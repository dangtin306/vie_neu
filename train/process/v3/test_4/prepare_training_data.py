#!/usr/bin/env python3
"""Build a training-only copy with long VAD-detected non-speech gaps shortened.

The source dataset is never edited. Short gaps and all detected speech are kept.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = HERE / "output/nghean_region_one_lora"
DEFAULT_SOURCE = WORK / "dataset"
DEFAULT_OUTPUT = WORK / "dataset_training_smooth_v2"
# Whisper audit found transcript mismatches/repeats in these source clips.
# Exclude only from this training copy; preserve every original WAV.
DEFAULT_EXCLUDE = ("0015_37_0062.wav", "0017_37_0077.wav", "0019_37_0082.wav", "0024_37_0123.wav")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--max-gap-sec", type=float, default=0.42,
                   help="only shorten VAD non-speech gaps longer than this")
    p.add_argument("--target-gap-sec", type=float, default=0.18,
                   help="replace a long non-speech gap with this much silence")
    p.add_argument("--speech-pad-ms", type=int, default=80)
    p.add_argument("--exclude", action="append", default=list(DEFAULT_EXCLUDE),
                   help="source WAV to omit from this training copy; repeatable")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def cap_long_vad_gaps(audio: np.ndarray, sr: int, timestamps,
                      max_gap_sec: float, target_gap_sec: float):
    if not timestamps:
        return audio, []
    segments = []
    for item in timestamps:
        start = max(0, min(len(audio), int(round(item["start"] * sr))))
        end = max(start, min(len(audio), int(round(item["end"] * sr))))
        if end > start:
            segments.append((start, end))
    if not segments:
        return audio, []

    pieces = []
    changes = []
    target = max(0, int(round(target_gap_sec * sr)))
    half_target = target // 2
    cursor = 0
    leading_gap = segments[0][0] / sr
    if leading_gap > max_gap_sec:
        pieces.append(np.zeros(half_target, dtype=np.float32))
        cursor = segments[0][0]
        changes.append({"edge": "leading", "original_gap_sec": round(leading_gap, 3),
                        "kept_silence_sec": round(half_target / sr, 3)})

    for (_, left_end), (right_start, _) in zip(segments, segments[1:]):
        gap_sec = (right_start - left_end) / sr
        if gap_sec <= max_gap_sec or right_start <= left_end:
            continue
        pieces.append(audio[cursor:left_end])
        pieces.append(np.zeros(target, dtype=np.float32))
        cursor = right_start
        changes.append({"edge": "internal", "start_sec": round(left_end / sr, 3),
                        "original_gap_sec": round(gap_sec, 3),
                        "kept_silence_sec": round(target / sr, 3)})

    trailing_gap = (len(audio) - segments[-1][1]) / sr
    if trailing_gap > max_gap_sec:
        pieces.append(audio[cursor:segments[-1][1]])
        pieces.append(np.zeros(half_target, dtype=np.float32))
        changes.append({"edge": "trailing", "start_sec": round(segments[-1][1] / sr, 3),
                        "original_gap_sec": round(trailing_gap, 3),
                        "kept_silence_sec": round(half_target / sr, 3)})
    else:
        pieces.append(audio[cursor:])
    if not changes:
        return audio, changes
    return np.concatenate(pieces).astype(np.float32, copy=False), changes

def main():
    args = parse_args()
    source = args.source_dir.expanduser().resolve()
    output = args.out_dir.expanduser().resolve()
    metadata = source / "metadata.csv"
    audio_dir = source / "raw_audio"
    if not metadata.is_file() or not audio_dir.is_dir():
        raise FileNotFoundError(f"Source dataset is incomplete: {source}")
    if args.max_gap_sec <= args.target_gap_sec or args.target_gap_sec < 0:
        raise ValueError("Require max-gap-sec > target-gap-sec >= 0")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace this clean copy")
        shutil.rmtree(output)
    out_audio = output / "raw_audio"
    out_audio.mkdir(parents=True)
    excluded = set(args.exclude or [])
    with metadata.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [(row[0].strip(), row[1].strip()) for row in csv.reader(f, delimiter="|")
                if len(row) >= 2 and row[0].strip() and row[1].strip()]

    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                  onnx=True, trust_repo=True)
    get_speech_timestamps = utils[0]
    out_rows = []
    report = {"source_dir": str(source), "output_dir": str(output),
              "max_gap_sec": args.max_gap_sec, "target_gap_sec": args.target_gap_sec,
              "excluded_files": [], "files": []}
    for name, text in rows:
        if name in excluded:
            report["excluded_files"].append(name)
            continue
        src = audio_dir / name
        if not src.is_file():
            raise FileNotFoundError(src)
        audio, sr = sf.read(src, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32)
        mono16 = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        timestamps = get_speech_timestamps(
            torch.from_numpy(np.asarray(mono16, dtype=np.float32)), model,
            sampling_rate=16000, threshold=0.42,
            min_silence_duration_ms=120, speech_pad_ms=args.speech_pad_ms,
            return_seconds=True,
        )
        cleaned, changes = cap_long_vad_gaps(
            audio, sr, timestamps, args.max_gap_sec, args.target_gap_sec)
        dst = out_audio / name
        sf.write(dst, cleaned, sr, subtype="PCM_16")
        out_rows.append((name, text))
        report["files"].append({
            "file_name": name, "duration_before_sec": round(len(audio) / sr, 3),
            "duration_after_sec": round(len(cleaned) / sr, 3),
            "compressed_gaps": changes,
        })
        if changes:
            print(f"{name}: shortened {len(changes)} long non-speech gap(s)")
    with (output / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|", lineterminator="\n")
        writer.writerows(out_rows)
    (output / "training_clean_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TRAINING DATA COPY: {len(out_rows)} clips; excluded {len(report['excluded_files'])}")
    print(f"dataset: {output}")
    print(f"report: {output / 'training_clean_report.json'}")


if __name__ == "__main__":
    main()
