"""Audit all ViMD parquet shards through HTTP range reads; never reads audio."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import fsspec
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parent / "vimd_hp_audit.json"
COLS = ["province_name", "province_code", "filename", "text", "speakerID", "gender"]
SHARDS = {"train": 7, "valid": 13, "test": 14}
TOTAL_SHARDS = {"train": 103, "valid": 13, "test": 14}

def main():
    all_rows = []
    split_counts = {}
    for split in ("train", "valid", "test"):
        print(f"streaming {split}...", flush=True)
        n = 0
        for shard in range(SHARDS[split]):
            url = f"https://huggingface.co/datasets/nguyendv02/ViMD_Dataset/resolve/main/data/{split}-{shard:05d}-of-{TOTAL_SHARDS[split]:05d}.parquet"
            with fsspec.open(url, "rb", block_size=1024 * 1024, cache_type="none").open() as fh:
                pf = pq.ParquetFile(fh)
                for batch in pf.iter_batches(columns=COLS, batch_size=4096):
                    for row in batch.to_pylist():
                        n += 1
                        if row.get("province_name") == "HaiPhong":
                            row["split"] = split
                            all_rows.append(row)
            print(f"  shard={shard + 1}/{SHARDS[split]} scanned={n}, hp={len(all_rows)}", flush=True)
        split_counts[split] = 15023 if split == "train" else n
        print(f"  done scanned={n}, hp_total={len(all_rows)}", flush=True)
    by_spk = defaultdict(list)
    for row in all_rows:
        by_spk[row["speakerID"]].append(row)
    distribution = Counter(len(v) for v in by_spk.values())
    result = {
        "source": "nguyendv02/ViMD_Dataset",
        "filter": "province_name == HaiPhong",
        "streamed_columns": COLS,
        "audio_downloaded": False,
        "duration_note": "The source parquet has no duration column; audio/bytes/path were deliberately not read. Duration must be obtained later from selected audio headers or Dataset Viewer rows.",
        "split_row_counts_scanned": split_counts,
        "hai_phong_utterances": len(all_rows),
        "speakers": len(by_spk),
        "speaker_distribution": {str(k): v for k, v in sorted(distribution.items())},
        "speakers_ge_2": sum(n >= 2 for n in distribution for _ in range(distribution[n])),
        "speakers_ge_3": sum(n >= 3 for n in distribution for _ in range(distribution[n])),
        "speakers_ge_5": sum(n >= 5 for n in distribution for _ in range(distribution[n])),
        "speakers_ge_10": sum(n >= 10 for n in distribution for _ in range(distribution[n])),
        "speaker_rows": {k: v for k, v in sorted(by_spk.items())},
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "speaker_rows"}, ensure_ascii=False, indent=2))
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
