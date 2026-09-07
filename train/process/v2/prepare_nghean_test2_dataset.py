from __future__ import annotations

import csv
import json
import math
import shutil
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "output" / "nghean_test2_accent_scale"
AUDIO = EXP / "audio"
AUDIT = ROOT / "vimd_na_audit.json"


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def audio_metrics(path: Path):
    try:
        with wave.open(str(path), "rb") as w:
            n, sr, ch = w.getnframes(), w.getframerate(), w.getnchannels()
            raw = w.readframes(n)
        if not raw or n == 0 or sr <= 0:
            raise ValueError("empty audio")
        y = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        y = y.reshape(-1, ch) if ch > 1 else y
        mono = y.mean(axis=1) if y.ndim == 2 else y
        duration = len(mono) / sr
        frame = max(1, int(sr * 0.02))
        energy = np.array([np.sqrt(np.mean(mono[i:i + frame] ** 2))
                           for i in range(0, len(mono), frame)])
        threshold = max(1e-5, float(np.max(energy)) * 0.02)
        silent = energy <= threshold
        spans = []
        i = 0
        while i < len(silent):
            if not silent[i]:
                i += 1
                continue
            j = i
            while j < len(silent) and silent[j]:
                j += 1
            spans.append((j - i) * frame / sr)
            i = j
        lead = 0.0
        while lead < len(silent) and silent[int(lead)]:
            lead += 1
        trail = 0
        while trail < len(silent) and silent[len(silent) - 1 - trail]:
            trail += 1
        internal = [x for x in spans if x > 0.04 and x < duration]
        internal_total = sum(internal)
        peak = float(np.max(np.abs(mono)))
        clip = float(np.mean(np.abs(mono) >= 0.999))
        status = "CLEAN"
        notes = []
        if clip > 0.001:
            status, notes = "SUSPECT", ["clipping_ratio>0.1%"]
        elif duration < 1.0 or duration > 60.0:
            status, notes = "SUSPECT", ["duration_outlier"]
        elif (lead * frame / sr) > 0.8 or (trail * frame / sr) > 0.8 or max(internal or [0]) > 1.0:
            status, notes = "SUSPECT", ["long_silence"]
        return {
            "duration": duration, "sample_rate": sr, "channels": ch,
            "peak": peak, "rms": float(np.sqrt(np.mean(mono ** 2))),
            "dc_offset": float(np.mean(mono)), "clipping_ratio": clip,
            "leading_silence": lead * frame / sr,
            "trailing_silence": trail * frame / sr,
            "internal_silence_ratio": internal_total / duration if duration else 0.0,
            "longest_internal_silence": max(internal or [0.0]),
            "music_flag": "UNASSESSED", "noise_flag": "UNASSESSED",
            "secondary_speech_flag": "UNASSESSED", "reverb_flag": "UNASSESSED",
            "cut_flag": "UNASSESSED", "qc_status": status,
            "notes": "; ".join(notes),
        }
    except Exception as e:
        return {"duration": "", "sample_rate": "", "channels": "", "peak": "",
                "rms": "", "dc_offset": "", "clipping_ratio": "",
                "leading_silence": "", "trailing_silence": "",
                "internal_silence_ratio": "", "longest_internal_silence": "",
                "music_flag": "UNASSESSED", "noise_flag": "UNASSESSED",
                "secondary_speech_flag": "UNASSESSED", "reverb_flag": "UNASSESSED",
                "cut_flag": "UNASSESSED", "qc_status": "BAD", "notes": str(e)}


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    all_rows = [r for p in audit["speaker_rows"].values() for r in p]
    eligible = {spk for spk, rows in audit["speaker_rows"].items() if len(rows) >= 2}
    eligible_rows = [r for r in all_rows if r["speakerID"] in eligible]
    exact = {"source": "nguyendv02/ViMD_Dataset", "province_code": 37,
             "province_name": "NgheAn", "total_utterances": len(all_rows),
             "total_speakers": len(audit["speaker_rows"]),
             "eligible_speakers_ge2": len(eligible),
             "eligible_utterances_ge2": len(eligible_rows), "splits": {}}
    for split in ("train", "valid", "test"):
        rs = [r for r in eligible_rows if r["split"] == split]
        exact["splits"][split] = {"speakers": len({r["speakerID"] for r in rs}), "utterances": len(rs)}
    exact["speaker_overlap_check"] = {
        "train_valid": not ({r["speakerID"] for r in eligible_rows if r["split"] == "train"} & {r["speakerID"] for r in eligible_rows if r["split"] == "valid"}),
        "train_test": not ({r["speakerID"] for r in eligible_rows if r["split"] == "train"} & {r["speakerID"] for r in eligible_rows if r["split"] == "test"}),
        "valid_test": not ({r["speakerID"] for r in eligible_rows if r["split"] == "valid"} & {r["speakerID"] for r in eligible_rows if r["split"] == "test"}),
    }
    (EXP / "dataset_audit_exact.json").write_text(json.dumps(exact, ensure_ascii=False, indent=2), encoding="utf-8")

    extracted = read_csv(EXP / "metadata_na_extracted.csv")
    manifest = []
    for r in extracted:
        p = Path(r["local_path"])
        manifest.append({"split": r["split"], "speakerID": r["speakerID"], "filename": r["filename"],
                         "source_id": f"{r['split']}/{r['filename']}", "transcript": r["text"],
                         "duration_metadata": r.get("duration_sec", ""), "local_path": str(p),
                         "exists": p.is_file(), "readable": False})
    write_csv(EXP / "nghean_eligible_audio_manifest.csv", manifest, list(manifest[0]))
    qc_rows = []
    for r in extracted:
        p = Path(r["local_path"])
        m = audio_metrics(p)
        qc_rows.append({"split": r["split"], "speakerID": r["speakerID"], "filename": r["filename"],
                        "transcript": r["text"], "local_path": str(p), **m})
    qc_fields = list(qc_rows[0])
    write_csv(EXP / "nghean_audio_qc.csv", qc_rows, qc_fields)
    qc_by_path = {r["local_path"]: r for r in qc_rows}
    for m in manifest:
        m["readable"] = qc_by_path.get(m["local_path"], {}).get("qc_status") != "BAD"
    write_csv(EXP / "nghean_eligible_audio_manifest.csv", manifest, list(manifest[0]))
    counts = Counter(r["qc_status"] for r in qc_rows)
    (EXP / "nghean_audio_qc_summary.json").write_text(json.dumps({"total": len(qc_rows), "counts": counts}, default=dict, ensure_ascii=False, indent=2), encoding="utf-8")

    # Review one technically usable clip per speaker, balanced across original splits.
    usable = [r for r in qc_rows if r["qc_status"] in {"CLEAN", "ACCEPTABLE"}]
    usable.sort(key=lambda r: (r["split"], r["speakerID"], r["filename"]))
    selected = []
    for r in usable:
        if r["speakerID"] not in {x["speakerID"] for x in selected}:
            selected.append(r)
        if len(selected) >= 30:
            break
    if len(selected) < 20:
        for r in usable:
            if r not in selected and len(selected) < 20:
                selected.append(r)
    review_dir = EXP / "accent_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    review_rows = []
    for i, r in enumerate(selected, 1):
        name = f"{r['speakerID']}__{r['filename']}"
        shutil.copy2(r["local_path"], review_dir / name)
        review_rows.append({"review_id": f"review_{i:03d}", "split": r["split"], "speakerID": r["speakerID"], "filename": r["filename"], "transcript": r["transcript"], "duration": r["duration"], "audio_quality": r["qc_status"], "accent_strength": "UNASSESSED", "notes": ""})
    write_csv(review_dir / "accent_review.csv", review_rows, list(review_rows[0]) if review_rows else ["review_id"])

    # Reuse extracted sequential same-speaker pairs, but emit the V2 schema and exclude BAD files.
    old_pairs = {s: read_csv(EXP / f"na_{s}_pairs.csv") for s in ("train", "valid", "test")}
    good = {r["local_path"] for r in qc_rows if r["qc_status"] in {"CLEAN", "ACCEPTABLE"}}
    pair_counts = {}
    pair_speakers = {}
    for split, pairs in old_pairs.items():
        out = []
        for i, p in enumerate(pairs, 1):
            if p["reference_path"] not in good or p["target_path"] not in good:
                continue
            out.append({"pair_id": f"na_v2_{split}_{i:04d}", "split": split, "speakerID": p["speakerID"],
                        "reference_filename": Path(p["reference_path"]).name, "reference_path": p["reference_path"], "reference_transcript": qc_by_path.get(p["reference_path"], {}).get("transcript", ""),
                        "target_filename": Path(p["target_path"]).name, "target_path": p["target_path"], "target_transcript": p["target_text"],
                        "reference_duration": p["reference_duration"], "target_duration": p["target_duration"],
                        "reference_qc_status": "CLEAN", "target_qc_status": "CLEAN"})
        write_csv(EXP / f"nghean_{split}_pairs_v2.csv", out, list(out[0]) if out else ["pair_id", "split", "speakerID"])
        pair_counts[split] = len(out); pair_speakers[split] = len({x["speakerID"] for x in out})
    overlap = not (set(pair_speakers) and False)
    split_sets = {s: {p["speakerID"] for p in read_csv(EXP / f"nghean_{s}_pairs_v2.csv")} for s in ("train", "valid", "test")}
    overlap = {"train_valid": sorted(split_sets["train"] & split_sets["valid"]), "train_test": sorted(split_sets["train"] & split_sets["test"]), "valid_test": sorted(split_sets["valid"] & split_sets["test"])}
    (EXP / "pilot_vs_v2_dataset.json").write_text(json.dumps({"pilot_train_pairs": 9, "v2_train_pairs": pair_counts["train"], "scale_factor_pairs": pair_counts["train"] / 9.0, "pilot_train_speakers": 9, "v2_train_speakers": pair_speakers["train"], "scale_factor_speakers": pair_speakers["train"] / 9.0}, ensure_ascii=False, indent=2), encoding="utf-8")
    ready = len(qc_rows) == 165 and not any(overlap.values()) and all(pair_counts[s] > 0 for s in ("train", "valid", "test"))
    (EXP / "prepare_summary.json").write_text(json.dumps({"exact": exact, "qc_counts": dict(counts), "accent_review_files": len(selected), "pair_counts": pair_counts, "pair_speakers": pair_speakers, "speaker_overlap": overlap, "ready_for_training": ready}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"extracted": len(qc_rows), "qc": dict(counts), "accent_review": len(selected), "pairs": pair_counts, "speakers": pair_speakers, "overlap": overlap, "READY_FOR_NGHEAN_TEST2_TRAINING": ready}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
