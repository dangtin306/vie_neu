#!/usr/bin/env python3
"""Stage 30 Nghệ An WAVs from test_2 and train one experimental VieNeu LoRA.

Uses the official VieNeu prepare_dataset.py and train_lora.py. The source WAVs
come from multiple speakers; this tests one shared regional adapter and is not
an official single-speaker fine-tune recipe.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import wave
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FT = ROOT / "source_code" / "audio_model" / "finetune"
SOURCE = ROOT / "train/process/v3/test_2/output/nghean_v3_turbo_region/dataset"
WORK = HERE / "output/nghean_v3_turbo_region_30"
DATA = WORK / "dataset"
AUDIO_OUT = DATA / "raw_audio"
TRAINING = WORK / "training"
RUN = "nghean_v3_turbo_region_30"
BASE = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
SELECTED_FILES = (
    "0002_37_0014.wav", "0004_37_0068.wav", "0005_37_0076.wav",
    "0006_37_0086.wav", "0008_37_0006.wav", "0009_37_0020.wav",
    "0010_37_0024.wav", "0012_37_0030.wav", "0015_37_0062.wav",
    "0017_37_0077.wav", "0019_37_0082.wav", "0021_37_0106.wav",
    "0023_37_0113.wav", "0024_37_0123.wav", "0025_37_0130.wav",
    "0027_37_0136.wav", "0029_37_0145.wav", "0031_37_0155.wav",
    "0033_37_0171.wav", "0036_37_0190.wav", "0039_37_0216.wav",
    "0041_37_0225.wav", "0042_37_0235.wav", "0043_37_0239.wav",
    "0044_37_0250.wav", "0048_37_0274.wav", "0050_37_0281.wav",
    "0055_37_0075.wav",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--train", action="store_true")
    g.add_argument("--all", action="store_true")
    p.add_argument("--overwrite-stage", action="store_true")
    return p.parse_args()


def read_source():
    meta = SOURCE / "metadata.csv"
    audio_dir = SOURCE / "raw_audio"
    if not meta.is_file() or not audio_dir.is_dir():
        raise FileNotFoundError(f"Expected source dataset at {SOURCE}")
    clips = []
    with meta.open("r", encoding="utf-8-sig", newline="") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.rstrip("\r\n").split("|", 1)
            if len(parts) != 2:
                raise ValueError(f"Bad source metadata line {line_no}: expected file|text")
            filename, text = (x.strip() for x in parts)
            # The staged upstream metadata has no speaker column; source WAV names
            # follow NNNN_37_NNNN.wav, where the final two fields identify speakers.
            stem_parts = Path(filename).stem.split("_")
            if len(stem_parts) < 3:
                raise ValueError(f"Cannot infer speaker ID from {filename}")
            speaker = "spk_" + "_".join(stem_parts[-2:])
            path = audio_dir / filename
            if not filename or not text or not speaker or not path.is_file():
                raise ValueError(f"Invalid source row {line_no}: {parts}")
            if "|" in text:
                raise ValueError(f"Transcript contains unsupported pipe at line {line_no}")
            with wave.open(str(path), "rb") as wav:
                duration = wav.getnframes() / wav.getframerate()
            if not 1.0 <= duration <= 20.0:
                raise ValueError(f"{filename} is {duration:.2f}s; official clips must be 1-20s")
            clips.append({"path": path, "filename": filename, "text": text,
                          "speaker": speaker, "duration": duration})
    return clips


def choose_selected(clips):
    by_name = {clip["filename"]: clip for clip in clips}
    missing = [name for name in SELECTED_FILES if name not in by_name]
    if missing:
        raise FileNotFoundError(f"Selected source WAVs missing from metadata: {missing}")
    return [by_name[name] for name in SELECTED_FILES]

def stage(overwrite):
    if DATA.exists():
        if not overwrite:
            raise FileExistsError(f"{DATA} already exists; pass --overwrite-stage to replace the staged dataset")
        shutil.rmtree(DATA)
    AUDIO_OUT.mkdir(parents=True, exist_ok=True)
    chosen = choose_selected(read_source())
    lines = []
    speakers = Counter()
    manifest = []
    for clip in chosen:
        name = clip["filename"]
        shutil.copy2(clip["path"], AUDIO_OUT / name)
        lines.append(f"{name}|{clip['text']}")
        speakers[clip["speaker"]] += 1
        manifest.append({
            "file_name": name,
            "source_file": clip["filename"],
            "source_speaker_id": clip["speaker"],
            "duration_sec": round(clip["duration"], 4),
            "text": clip["text"],
        })
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    minutes = sum(x["duration"] for x in chosen) / 60
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "source_manifest.json").write_text(json.dumps({
        "experiment": "one shared regional LoRA from 28 curated clips",
        "source_dataset": str(SOURCE),
        "clip_count": len(chosen),
        "unique_source_speakers": len(speakers),
        "total_minutes": round(minutes, 3),
        "upstream_note": "Uses official scripts and one LoRA, but source clips are multi-speaker; upstream guide requires one real speaker per LoRA.",
        "rows": manifest,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"STAGED: {len(chosen)} clips | {len(speakers)} source speakers | {minutes:.2f} min")
    if minutes < 10:
        print("WARNING: below upstream 10-30 minute guidance; treat as a pilot.")
    print(f"dataset: {DATA}")
    print(f"metadata: {DATA / 'metadata.csv'}")


def run(cmd):
    print("run:", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=ROOT, check=True)


def prepare():
    script = FT / "prepare_dataset.py"
    out = DATA / "train.parquet"
    if not script.is_file():
        raise FileNotFoundError(script)
    run([sys.executable, "-u", script, "--dataset-dir", DATA, "--out", out,
         "--base", BASE, "--min-sec", "1.0", "--max-sec", "20.0"])
    if not out.is_file():
        raise RuntimeError(f"Preparation did not create {out}")
    print(f"PREPARE COMPLETE: {out}")


def train():
    data = DATA / "train.parquet"
    script = FT / "train_lora.py"
    if not data.is_file():
        raise FileNotFoundError(f"Missing {data}; run --prepare first.")
    if not script.is_file():
        raise FileNotFoundError(script)
    TRAINING.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-u", script, "--data", data, "--run", RUN,
         "--output-dir", TRAINING, "--base", BASE, "--subfolder", "update",
         "--r", "16", "--alpha", "32", "--dropout", "0.05",
         "--target", "backbone", "--batch-size", "1", "--grad-accum", "4",
         "--lr", "0.0002", "--epochs", "3", "--eval-ratio", "0.2",
         "--eval-every", "5", "--save-every", "5", "--log-every", "2",
         "--max-length", "1024", "--seed", "42", "--num-workers", "0", "--merge"])
    print(f"TRAIN COMPLETE: {TRAINING / RUN}")


def main():
    args = parse_args()
    if args.prepare or args.all:
        stage(args.overwrite_stage)
        prepare()
    if args.train or args.all:
        train()


if __name__ == "__main__":
    main()
