#!/usr/bin/env python3
"""Train one shared regional LoRA from curated multi-speaker clips.

This is an experimental extension of VieNeu's documented single-speaker recipe.
Each dataset row keeps the speaker embedding extracted from its own source WAV;
all rows update one shared LoRA. Inference should use one chosen speaker anchor.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FT = ROOT / "source_code" / "audio_model" / "finetune"
SOURCE = ROOT / "train/process/v3/test_3/output/nghean_v3_turbo_region_30/dataset"
WORK = HERE / "output/nghean_region_one_lora"
DATA = WORK / "dataset"
AUDIO_OUT = DATA / "raw_audio"
TRAINING = WORK / "training"
RUN = "nghean_region_one_lora"
BASE = "pnnbao-ump/VieNeu-TTS-v3-Turbo"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--train", action="store_true")
    modes.add_argument("--all", action="store_true")
    p.add_argument("--overwrite-stage", action="store_true",
                   help="replace only test_4's copied dataset")
    p.add_argument("--epochs", type=float, default=3.0,
                   help="official upstream baseline; ignored when --max-steps is set")
    p.add_argument("--max-steps", type=int, default=0,
                   help="optional step override; 0 uses --epochs")
    p.add_argument("--run-name", default=RUN,
                   help="separate output name for comparing training runs")
    return p.parse_args()


def source_rows():
    meta = SOURCE / "metadata.csv"
    audio_dir = SOURCE / "raw_audio"
    if not meta.is_file() or not audio_dir.is_dir():
        raise FileNotFoundError(f"Curated source dataset not found: {SOURCE}")
    rows = []
    with meta.open("r", encoding="utf-8-sig", newline="") as f:
        for line_no, row in enumerate(csv.reader(f, delimiter="|"), 1):
            if len(row) != 2:
                raise ValueError(f"Bad metadata row {line_no}; expected file|text")
            name, text = (part.strip() for part in row)
            wav_path = audio_dir / name
            if not name or not text or not wav_path.is_file():
                raise ValueError(f"Invalid source row {line_no}: {row}")
            with wave.open(str(wav_path), "rb") as wav:
                duration = wav.getnframes() / wav.getframerate()
            if not 1.0 <= duration <= 20.0:
                raise ValueError(f"{name} duration {duration:.2f}s is outside 1-20s")
            fields = Path(name).stem.split("_")
            speaker_id = "spk_" + "_".join(fields[-2:]) if len(fields) >= 3 else "unknown"
            rows.append({"name": name, "text": text, "path": wav_path,
                         "duration": duration, "speaker_id": speaker_id})
    if not rows:
        raise ValueError("Curated source dataset is empty")
    return rows


def stage(overwrite: bool):
    rows = source_rows()
    if DATA.exists():
        if not overwrite:
            raise FileExistsError(f"{DATA} exists; pass --overwrite-stage to replace it")
        shutil.rmtree(DATA)
    AUDIO_OUT.mkdir(parents=True, exist_ok=True)
    metadata_lines = []
    manifest_rows = []
    for row in rows:
        shutil.copy2(row["path"], AUDIO_OUT / row["name"])
        metadata_lines.append(f"{row['name']}|{row['text']}")
        manifest_rows.append({"file_name": row["name"], "source_speaker_id": row["speaker_id"],
                              "duration_sec": round(row["duration"], 4), "text": row["text"]})
    (DATA / "metadata.csv").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
    minutes = sum(row["duration"] for row in rows) / 60
    speakers = {row["speaker_id"] for row in rows}
    (WORK / "source_manifest.json").write_text(json.dumps({
        "experiment": "many-speaker regional data -> one shared LoRA -> one anchored output voice",
        "source_dataset": str(SOURCE), "clip_count": len(rows),
        "total_minutes": round(minutes, 3),
        "conditioning": "prepare_dataset.py extracts one speaker_embedding per clip; training uses no reference codes",
        "upstream_note": "Experimental multi-speaker use; official README documents one speaker per LoRA.",
        "rows": manifest_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"STAGED: {len(rows)} clips | {len(speakers)} source speakers | {minutes:.2f} min; original WAV names preserved")
    if len(rows) < 100 or minutes < 10:
        print("NOTE: smaller than the upstream one-voice data guidance (100-300 clips, 10-30 min); this multi-speaker regional run remains experimental.")
    print(f"dataset: {DATA}")
    print(f"metadata: {DATA / 'metadata.csv'}")


def run(cmd):
    print("run:", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=ROOT, check=True)


def prepare():
    script = FT / "prepare_dataset.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    run([sys.executable, "-u", script, "--dataset-dir", DATA,
         "--out", DATA / "train.parquet", "--base", BASE,
         "--min-sec", "1.0", "--max-sec", "20.0"])
    print(f"PREPARE COMPLETE: {DATA / 'train.parquet'}")


def train(max_steps: int, epochs: float, run_name: str):
    data = DATA / "train.parquet"
    script = FT / "train_lora.py"
    if not data.is_file():
        raise FileNotFoundError(f"Missing {data}; run --prepare first")
    run([sys.executable, "-u", script,
         "--data", data, "--run", run_name, "--output-dir", TRAINING,
         "--base", BASE, "--subfolder", "update",
         "--r", "16", "--alpha", "32", "--dropout", "0.05",
         "--target", "backbone", "--batch-size", "1", "--grad-accum", "4",
         "--lr", "0.0002",
         "--eval-ratio", "0.2", "--eval-every", "3", "--save-every", "3",
         "--log-every", "2", "--max-length", "1024", "--seed", "42",
         "--num-workers", "0", "--merge"] +
         (["--max-steps", str(max_steps)] if max_steps > 0 else ["--epochs", str(epochs)]))
    print(f"TRAIN COMPLETE: {TRAINING / run_name}")
    print(f"MERGED MODEL: {TRAINING / run_name / 'merged'}")


def main():
    args = parse_args()
    if args.max_steps < 0 or args.epochs <= 0:
        raise ValueError("--max-steps must be nonnegative and --epochs positive")
    if args.prepare or args.all:
        stage(args.overwrite_stage)
        prepare()
    if args.train or args.all:
        train(args.max_steps, args.epochs, args.run_name)


if __name__ == "__main__":
    main()
