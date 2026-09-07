"""Extract eligible HaiPhong audio from ViMD parquet and build ref/target pairs.

Only parquet shards containing eligible rows have their audio column read. No
training or model code is imported here.
"""
from __future__ import annotations

import csv
import json
import math
import wave
from collections import defaultdict
from pathlib import Path

import fsspec
import numpy as np
import pyarrow.parquet as pq

TRAIN_DIR = Path(__file__).resolve().parent
AUDIT = TRAIN_DIR / "vimd_hp_audit.json"
OUT_ROOT = TRAIN_DIR / "data" / "dataset_haiphong"
EXTRACTED_CSV = TRAIN_DIR / "metadata_hp_extracted.csv"
SHARDS = {"train": 103, "valid": 13, "test": 14}
COLS_META = ["province_name", "province_code", "filename", "text", "speakerID", "gender"]
COLS_AUDIO = COLS_META + ["audio"]


def parquet_url(split: str, index: int) -> str:
    return f"https://huggingface.co/datasets/nguyendv02/ViMD_Dataset/resolve/main/data/{split}-{index:05d}-of-{SHARDS[split]:05d}.parquet"


def eligible_rows():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    eligible = {}
    for speaker, rows in audit["speaker_rows"].items():
        if len(rows) >= 2:
            for row in rows:
                row = dict(row)
                row["_key"] = (row["split"], row["filename"])
                eligible[row["_key"]] = row
    return eligible


def find_shard_rows(eligible):
    found = defaultdict(list)
    for split, n_shards in SHARDS.items():
        # The completed full audit located all eligible train rows in shards
        # 5 and 6; valid/test are scanned completely.
        shard_ids = (5, 6) if split == "train" else range(n_shards)
        for idx in shard_ids:
            url = parquet_url(split, idx)
            with fsspec.open(url, "rb", block_size=1024 * 1024, cache_type="none").open() as fh:
                pf = pq.ParquetFile(fh)
                for batch in pf.iter_batches(columns=COLS_META, batch_size=4096):
                    for row in batch.to_pylist():
                        key = (split, row["filename"])
                        if key in eligible:
                            merged = dict(eligible[key])
                            merged.update({k: row.get(k) for k in COLS_META})
                            found[(split, idx)].append(merged)
            if found.get((split, idx)):
                print(f"metadata {split} shard {idx}: {len(found[(split, idx)])} eligible rows", flush=True)
    return found


def pcm_stats(raw: bytes, sample_width: int, channels: int, rate: int):
    if sample_width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        x = b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)
        x = np.where(x & 0x800000, x - 0x1000000, x)
        audio = x.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported PCM sample width: {sample_width}")
    if audio.size == 0:
        raise ValueError("empty audio")
    return {
        "sample_rate": rate,
        "duration_sec": float(audio.size / channels / rate),
        "channels": channels,
        "peak": float(np.max(np.abs(audio))),
        "rms": float(math.sqrt(np.mean(audio * audio))),
    }


def qc_wav(path: Path):
    try:
        with wave.open(str(path), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            stats = pcm_stats(raw, wf.getsampwidth(), wf.getnchannels(), wf.getframerate())
        reasons = []
        if not 4.0 <= stats["duration_sec"] <= 15.0:
            # Duration is a preference for this pilot, not an automatic reject.
            reasons.append("duration_preference_outside_4_15s")
        if stats["peak"] >= 0.999:
            reasons.append("possible_clipping")
        hard_fail = any(reason == "possible_clipping" for reason in reasons)
        if hard_fail:
            stats["qc_status"] = "flag:" + ",".join(reasons)
        elif reasons:
            stats["qc_status"] = "pass:" + ",".join(reasons)
        else:
            stats["qc_status"] = "pass"
        return stats
    except Exception as exc:
        return {"sample_rate": "", "duration_sec": "", "channels": "", "peak": "", "rms": "", "qc_status": "error:" + str(exc)}


def extract(found):
    rows = []
    for (split, idx), wanted in found.items():
        wanted_names = {r["filename"] for r in wanted}
        url = parquet_url(split, idx)
        existing = [OUT_ROOT / split / r["speakerID"] / r["filename"] for r in wanted]
        if all(path.is_file() for path in existing):
            for row, path in zip(wanted, existing):
                rows.append({"split": split, "speakerID": row["speakerID"], "filename": row["filename"], "text": row["text"], "local_path": str(path), **qc_wav(path)})
            print(f"reuse local audio {split} shard {idx} ({len(existing)} rows)", flush=True)
            continue
        print(f"reading audio {split} shard {idx} ({len(wanted_names)} rows)", flush=True)
        with fsspec.open(url, "rb", block_size=1024 * 1024, cache_type="none").open() as fh:
            table = pq.read_table(fh, columns=COLS_AUDIO)
        for row in table.to_pylist():
            if row["filename"] not in wanted_names or row.get("province_name") != "HaiPhong":
                continue
            audio = row.get("audio") or {}
            raw = audio.get("bytes")
            if not raw:
                raise RuntimeError(f"No embedded audio bytes for {split}/{row['filename']}; path={audio.get('path')}")
            path = OUT_ROOT / split / row["speakerID"] / row["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            stats = qc_wav(path)
            rows.append({"split": split, "speakerID": row["speakerID"], "filename": row["filename"], "text": row["text"], "local_path": str(path), **stats})
    rows.sort(key=lambda r: (r["split"], r["speakerID"], r["filename"]))
    return rows


def write_csv(path: Path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def make_pairs(rows):
    # Duration-out-of-preference remains usable; clipping/read errors do not.
    valid = [r for r in rows if str(r["qc_status"]).startswith("pass")]
    by = defaultdict(list)
    for row in valid:
        by[(row["split"], row["speakerID"])].append(row)
    pairs = []
    for (split, speaker), clips in sorted(by.items()):
        clips.sort(key=lambda r: r["filename"])
        pair_count = 1 if len(clips) == 2 else len(clips) - 1
        for i in range(pair_count):
            ref = clips[i]
            target = clips[i + 1]
            if ref["local_path"] == target["local_path"] or ref["speakerID"] != target["speakerID"]:
                raise RuntimeError("invalid reference/target pair")
            pairs.append({"speakerID": speaker, "reference_path": ref["local_path"], "target_path": target["local_path"], "target_text": target["text"], "reference_duration": ref["duration_sec"], "target_duration": target["duration_sec"], "split": split})
    return pairs


def main():
    eligible = eligible_rows()
    print(f"eligible rows from audit: {len(eligible)}", flush=True)
    found = find_shard_rows(eligible)
    rows = extract(found)
    fields = ["split", "speakerID", "filename", "text", "local_path", "duration_sec", "sample_rate", "channels", "peak", "rms", "qc_status"]
    write_csv(EXTRACTED_CSV, rows, fields)
    pairs = make_pairs(rows)
    pair_fields = ["speakerID", "reference_path", "target_path", "target_text", "reference_duration", "target_duration", "split"]
    for split in ("train", "valid", "test"):
        split_pairs = [p for p in pairs if p["split"] == split]
        write_csv(TRAIN_DIR / f"{split}_pairs.csv", split_pairs, pair_fields)
    summary = {"eligible_audit_rows": len(eligible), "extracted": len(rows), "by_split": {}, "qc_pass": sum(str(r["qc_status"]).startswith("pass") for r in rows), "qc_nonpass": sum(not str(r["qc_status"]).startswith("pass") for r in rows), "pairs": {s: sum(p["split"] == s for p in pairs) for s in ("train", "valid", "test")}}
    for split in ("train", "valid", "test"):
        subset = [r for r in rows if r["split"] == split]
        summary["by_split"][split] = {"clips": len(subset), "speakers": len({r["speakerID"] for r in subset}), "qc_pass": sum(str(r["qc_status"]).startswith("pass") for r in subset)}
    (TRAIN_DIR / "test13_extract_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

