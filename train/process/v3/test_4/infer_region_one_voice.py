#!/usr/bin/env python3
"""Smooth long-form inference with one anchored voice and controlled chunk joins."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE_ROOT = ROOT / "source_code" / "audio_model"
WORK = HERE / "output/nghean_region_one_lora"
RUN = "nghean_region_one_lora_90"
DEFAULT_MERGED = WORK / "training" / RUN / "merged"
DEFAULT_REFERENCE = ROOT / "train/process/v3/test_3/output/nghean_v3_turbo_region_30/dataset/raw_audio/0017_37_0077.wav"
DEFAULT_OUTPUT = WORK / "demo_smooth_90"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--text", action="append", required=True,
                   help="text to synthesize; repeat the option for multiple WAVs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.45)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--top-p", type=float, default=0.85)
    p.add_argument("--repetition-penalty", type=float, default=1.4)
    p.add_argument("--max-new-frames", type=int, default=1800)
    p.add_argument("--max-chars", type=int, default=160)
    p.add_argument("--minor-pause", type=float, default=0.16)
    p.add_argument("--sentence-pause", type=float, default=0.28)
    p.add_argument("--paragraph-pause", type=float, default=0.42)
    p.add_argument("--no-ref-codes", action="store_true",
                   help="comparison only; omit anchor audio codes")
    return p.parse_args()


def resolve(path: Path) -> Path:
    return path.expanduser() if path.is_absolute() else ROOT / path


def join_chunks(chunks, gaps, sr, pauses):
    from vieneu_utils.core_utils import pause_pad_samples

    parts = []
    previous_audio = None
    for i, audio in enumerate(chunks):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not audio.size:
            continue
        if previous_audio is not None:
            pause = pauses.get(gaps[i - 1], pauses["sentence"])
            pad = pause_pad_samples(previous_audio, audio, sr, pause)
            if pad:
                parts.append(np.zeros(pad, dtype=np.float32))
        parts.append(audio)
        previous_audio = audio
    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)


def main():
    args = parse_args()
    if args.max_chars < 40:
        raise ValueError("--max-chars must be at least 40")
    merged = resolve(args.merged)
    reference = resolve(args.reference)
    output_dir = resolve(args.output_dir)
    if not (merged / "config.json").is_file():
        raise FileNotFoundError(f"Merged model not found: {merged}")
    if not reference.is_file():
        raise FileNotFoundError(f"Reference WAV not found: {reference}")
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
    from vieneu import Vieneu
    from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass

    output_dir.mkdir(parents=True, exist_ok=True)
    tts = Vieneu(mode="v3turbo", backbone_repo=str(merged), device="auto",
                 backend="pytorch", babble_retries=2)
    use_ref_codes = not args.no_ref_codes
    pauses = {
        "minor": args.minor_pause,
        "sentence": args.sentence_pause,
        "para": args.paragraph_pause,
    }
    reports = []
    try:
        for index, text in enumerate(args.text, 1):
            chunks, gaps = normalize_to_chunks_v3_with_gaps(
                text, max_chars=args.max_chars)
            print(f"Generating {index}/{len(args.text)}: {len(chunks)} chunks; "
                  f"batch_size=1; use_ref_codes={use_ref_codes}; pauses={pauses}",
                  flush=True)
            wav_chunks = []
            for chunk_no, chunk in enumerate(chunks, 1):
                print(f"  chunk {chunk_no}/{len(chunks)}", flush=True)
                audio = tts.infer(
                    chunk,
                    ref_audio=str(reference),
                    use_ref_codes=use_ref_codes,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    max_new_frames=args.max_new_frames,
                    max_chars=args.max_chars,
                    batch_size=1,
                    apply_watermark=False,
                )
                wav_chunks.append(audio)
            audio = join_chunks(wav_chunks, gaps, tts.sample_rate, pauses)
            destination = output_dir / f"region_voice_{index:02d}.wav"
            tts.save(audio, str(destination))
            duration = len(audio) / float(tts.sample_rate)
            print(f"saved {destination} duration={duration:.2f}s", flush=True)
            reports.append({
                "text": text, "wav": str(destination),
                "duration_sec": round(duration, 4), "chunks": chunks,
                "gaps": gaps, "sampling": {
                    "temperature": args.temperature, "top_k": args.top_k,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "batch_size": 1,
                },
                "use_ref_codes": use_ref_codes, "pauses_sec": pauses,
            })
    finally:
        close = getattr(tts, "close", None)
        if callable(close):
            close()
    (output_dir / "demo_report.json").write_text(json.dumps({
        "merged_model": str(merged), "reference": str(reference),
        "use_ref_codes": use_ref_codes, "results": reports,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DEMO COMPLETE: {output_dir}")


if __name__ == "__main__":
    main()
