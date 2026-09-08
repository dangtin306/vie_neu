"""Generate a small Base vs official v3 Turbo fine-tune demo.

The fine-tuned side is loaded from the official merged model produced by
``train_v3_turbo_30.py``.  This keeps the test compatible with the v3 Turbo
SDK and avoids the old v2 PEFT/NeuCodec inference path.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


def configure_console() -> None:
    """Keep Vietnamese logs printable in Windows consoles and Ubuntu shells."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[4]
SOURCE_ROOT = PROJECT_ROOT / "source_code" / "audio_model"
WORK_ROOT = HERE / "output" / "nghean_v3_turbo_30"
STAGE_DATASET = WORK_ROOT / "dataset"
DEFAULT_MERGED = WORK_ROOT / "training" / "nghean_v3_turbo_30" / "merged"
BASE_MODEL = "pnnbao-ump/VieNeu-TTS-v3-Turbo"

DEFAULT_TEXTS = [
    "Hôm nay tôi muốn thử một câu tiếng Việt mới.",
    "Giọng nói cần giữ được nhịp tự nhiên và cách nhấn rõ ràng.",
    "Đây là bản kiểm tra VieNeu-TTS v3 Turbo sau khi fine-tune.",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, default=None)
    ap.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    ap.add_argument("--output-dir", type=Path, default=WORK_ROOT / "demo")
    ap.add_argument("--text", action="append", dest="texts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.2)
    ap.add_argument("--max-new-frames", type=int, default=300)
    ap.add_argument(
        "--model",
        choices=("base", "finetuned", "both"),
        default="both",
    )
    return ap.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def choose_reference(explicit: Path | None) -> tuple[Path, str, str]:
    if explicit is not None:
        ref = resolve(explicit)
        if not ref.is_file():
            raise FileNotFoundError(f"Reference not found: {ref}")
        return ref, "manual", ""

    metadata = STAGE_DATASET / "metadata.csv"
    if not metadata.is_file():
        raise FileNotFoundError(
            f"No staged metadata. Run train_v3_turbo_30.py --prepare first, "
            f"or pass --reference."
        )
    with metadata.open("r", encoding="utf-8", newline="") as f:
        row = next(csv.reader(f, delimiter="|"), None)
    if not row or len(row) < 2:
        raise RuntimeError(f"Invalid official metadata: {metadata}")
    ref = STAGE_DATASET / "raw_audio" / row[0]
    if not ref.is_file():
        raise FileNotFoundError(f"Staged reference not found: {ref}")
    return ref, row[2] if len(row) > 2 else "nghean_30", row[1]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_tts(backbone: str | Path):
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
    from vieneu import Vieneu

    # v3 Turbo automatically selects PyTorch/CUDA or ONNX/CPU.  A merged
    # fine-tuned model must be tested on CUDA because official v3 merge output
    # is not consumed by the SDK's CPU/ONNX graph.
    return Vieneu(
        mode="v3turbo",
        backbone_repo=str(backbone),
        device="auto",
        backend="auto",
    )


def generate_one(
    label: str,
    backbone: str | Path,
    reference: Path,
    texts: list[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, Any]]:
    print(f"loading {label}: {backbone}", flush=True)
    tts = load_tts(backbone)
    results = []
    try:
        for index, text in enumerate(texts, start=1):
            seed = args.seed + index * 1009
            seed_everything(seed)
            destination = output_dir / label / f"sentence_{index:02d}.wav"
            destination.parent.mkdir(parents=True, exist_ok=True)
            print(f"generate {label} {index}/{len(texts)}", flush=True)
            audio = tts.infer(
                text,
                ref_audio=str(reference),
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_new_frames=args.max_new_frames,
                apply_watermark=False,
            )
            tts.save(audio, str(destination))
            duration = float(len(audio)) / float(tts.sample_rate)
            results.append(
                {
                    "label": label,
                    "index": index,
                    "text": text,
                    "seed": seed,
                    "wav": str(destination),
                    "duration_sec": round(duration, 4),
                }
            )
            print(f"saved {destination} duration={duration:.3f}s", flush=True)
    finally:
        close = getattr(tts, "close", None)
        if callable(close):
            close()
    return results


def main() -> None:
    configure_console()
    args = parse_args()
    reference, speaker, ref_text = choose_reference(args.reference)
    merged = resolve(args.merged)
    output_dir = resolve(args.output_dir)
    texts = args.texts or DEFAULT_TEXTS
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reference.wav").write_bytes(reference.read_bytes())

    config = {
        "base_model": BASE_MODEL,
        "merged_model": str(merged),
        "speaker": speaker,
        "reference": str(reference),
        "reference_text": ref_text,
        "seed_base": args.seed,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_new_frames": args.max_new_frames,
        "texts": texts,
        "models_requested": args.model,
    }
    reports: list[dict[str, Any]] = []

    if args.model in ("base", "both"):
        reports.extend(
            generate_one(
                "base",
                BASE_MODEL,
                reference,
                texts,
                args,
                output_dir,
            )
        )
    if args.model in ("finetuned", "both"):
        if not merged.is_dir():
            raise FileNotFoundError(
                f"Merged v3 model not found: {merged}. "
                "Finish training with --merge first."
            )
        reports.extend(
            generate_one(
                "v3_lora",
                merged,
                reference,
                texts,
                args,
                output_dir,
            )
        )

    (output_dir / "demo_report.json").write_text(
        json.dumps({"config": config, "results": reports}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"DEMO COMPLETE: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
