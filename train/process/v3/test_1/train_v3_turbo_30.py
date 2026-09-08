"""Prepare 30 Nghệ An clips and run the official VieNeu-TTS v3 Turbo LoRA flow.

This is intentionally a small, portable runner.  It does not reuse the old
v2 NeuCodec/0.3B trainer.  The official v3 fine-tune scripts are used when
present in ``source_code/audio_model/finetune``; otherwise the exact files from
the official GitHub repository are cached under this experiment directory.

Examples
--------
Windows PowerShell / cmd and Ubuntu:

    python -u train/process/v3/test_1/train_v3_turbo_30.py --prepare
    python -u train/process/v3/test_1/train_v3_turbo_30.py --train
    python -u train/process/v3/test_1/train_v3_turbo_30.py --all

``--prepare`` is CPU/ONNX and works without CUDA.  Official v3 LoRA training
requires a CUDA GPU; the script stops with a clear message on CPU instead of
trying a memory-heavy CPU run.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any


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
# train/process/v3/test_1 -> repository root is four levels above the file's
# directory (parents[3]); using parents[4] drops the `vie_neu` directory on
# Ubuntu and makes metadata paths resolve to /root/media_tech_ai/train/.
PROJECT_ROOT = HERE.parents[3]
SOURCE_ROOT = PROJECT_ROOT / "source_code" / "audio_model"
WORK_ROOT = HERE / "output" / "nghean_v3_turbo_30"
STAGE_DATASET = WORK_ROOT / "dataset"
STAGE_AUDIO = STAGE_DATASET / "raw_audio"
TRAINING_ROOT = WORK_ROOT / "training"
RUN_NAME = "nghean_v3_turbo_30"
BASE_MODEL = "pnnbao-ump/VieNeu-TTS-v3-Turbo"

SOURCE_METADATA_PRIMARY = (
    PROJECT_ROOT
    / "train"
    / "output"
    / "nghean_test2_accent_scale"
    / "metadata_na_extracted.csv"
)
SOURCE_METADATA_FALLBACK = (
    PROJECT_ROOT
    / "train"
    / "output"
    / "nghean_test2_accent_scale"
    / "nghean_eligible_audio_manifest.csv"
)
# The old extracted metadata is not part of Git.  New Ubuntu clones commonly
# have only the eligible manifest, so choose it automatically when present.
SOURCE_METADATA = (
    SOURCE_METADATA_PRIMARY
    if SOURCE_METADATA_PRIMARY.is_file()
    else SOURCE_METADATA_FALLBACK
)
SOURCE_PAIR_METADATA = (
    PROJECT_ROOT
    / "train"
    / "output"
    / "nghean_test2_accent_scale"
    / "nghean_train_pairs_v2.csv"
)
SOURCE_AUDIO_ROOT = PROJECT_ROOT / "train" / "data" / "dataset_nghean"

OFFICIAL_BASE_URL = (
    "https://raw.githubusercontent.com/pnnbao97/VieNeu-TTS/main/finetune/"
)
OFFICIAL_FILES = {
    "prepare_dataset.py": "prepare_dataset.py",
    "train_lora.py": "train_lora.py",
    "merge_lora.py": "merge_lora.py",
    "vieneu_lora/__init__.py": "vieneu_lora/__init__.py",
    "vieneu_lora/data.py": "vieneu_lora/data.py",
    "vieneu_lora/model.py": "vieneu_lora/model.py",
    "vieneu_lora/lora.py": "vieneu_lora/lora.py",
    "vieneu_lora/utils.py": "vieneu_lora/utils.py",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--train", action="store_true")
    action.add_argument("--all", action="store_true")

    ap.add_argument("--source-metadata", type=Path, default=SOURCE_METADATA)
    ap.add_argument("--source-audio-root", type=Path, default=SOURCE_AUDIO_ROOT)
    ap.add_argument("--source-pairs", type=Path, default=SOURCE_PAIR_METADATA)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument(
        "--max-sec",
        type=float,
        default=30.0,
        help="Maximum clip duration passed to official v3 preparation.",
    )
    ap.add_argument("--overwrite-stage", action="store_true")

    # Official v3 defaults are r16/a32.  They remain CLI options so the pilot
    # can be repeated without editing code, but no old v2 target is assumed.
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--target", default="backbone")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-grad-checkpoint", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    return ap.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    # Older manifests call the transcript column `transcript`; the official
    # v3 preparation flow expects `text`.
    for row in rows:
        if not row.get("text", "").strip() and row.get("transcript", "").strip():
            row["text"] = row["transcript"].strip()
    return rows


def find_source_audio(row: dict[str, str], audio_root: Path) -> Path:
    local = Path(row.get("local_path", "")).expanduser()
    if local.is_file():
        return local

    split = row.get("split", "train").strip()
    speaker = row.get("speakerID", "").strip()
    filename = row.get("filename", "").strip()
    candidates = [audio_root / split / speaker / filename]
    candidates.extend(audio_root.glob(f"{split}/{speaker}/*{filename}"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Last resort for a relocated dataset: search only inside the declared
    # audio root, never the whole disk.
    matches = list(audio_root.rglob(filename)) if filename else []
    for candidate in matches:
        if speaker and speaker not in candidate.parts:
            continue
        return candidate
    raise FileNotFoundError(
        f"Audio not found for {speaker}/{filename}. Checked metadata path and {audio_root}."
    )


def select_rows(
    metadata_path: Path,
    pairs_path: Path,
    audio_root: Path,
    limit: int,
) -> list[tuple[dict[str, str], Path]]:
    rows = read_csv(metadata_path)
    by_key = {
        (r.get("speakerID", "").strip(), r.get("filename", "").strip()): r
        for r in rows
        if r.get("split", "train").strip().lower() == "train"
    }

    # Pair order gives us reference/target diversity and, where available,
    # same-speaker peers for the official v3 data loader.
    ordered_keys: list[tuple[str, str]] = []
    if pairs_path.is_file():
        for pair in read_csv(pairs_path):
            if pair.get("split", "").strip().lower() != "train":
                continue
            speaker = pair.get("speakerID", "").strip()
            for field in ("reference_filename", "target_filename"):
                key = (speaker, pair.get(field, "").strip())
                if key in by_key and key not in ordered_keys:
                    ordered_keys.append(key)

    for key in by_key:
        if key not in ordered_keys:
            ordered_keys.append(key)

    selected: list[tuple[dict[str, str], Path]] = []
    for key in ordered_keys:
        if len(selected) >= limit:
            break
        row = by_key[key]
        selected.append((row, find_source_audio(row, audio_root)))
    if len(selected) < limit:
        raise RuntimeError(
            f"Only {len(selected)} usable train WAVs found; required {limit}."
        )
    return selected


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "existing"
    try:
        os.link(source, destination)
        return "hardlink"
    except (OSError, NotImplementedError):
        shutil.copy2(source, destination)
        return "copy"


def prepare_stage(args: argparse.Namespace) -> Path:
    metadata_path = resolve(args.source_metadata)
    audio_root = resolve(args.source_audio_root)
    pairs_path = resolve(args.source_pairs)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Audio root not found: {audio_root}")

    if args.overwrite_stage and STAGE_DATASET.exists():
        shutil.rmtree(STAGE_DATASET)
    STAGE_AUDIO.mkdir(parents=True, exist_ok=True)
    selected = select_rows(metadata_path, pairs_path, audio_root, args.limit)

    manifest: list[dict[str, Any]] = []
    metadata_lines = []
    for index, (row, source) in enumerate(selected, start=1):
        text = row.get("text", "").strip()
        speaker = row.get("speakerID", "").strip()
        if not text or not speaker:
            raise ValueError(f"Missing text/speaker in source row: {row}")
        if "|" in text:
            raise ValueError(
                f"Official metadata format cannot preserve '|' in transcript: {source}"
            )
        staged_name = f"{index:03d}_{source.name}"
        staged_path = STAGE_AUDIO / staged_name
        materialization = link_or_copy(source, staged_path)
        metadata_lines.append(f"{staged_name}|{text}|{speaker}")
        manifest.append(
            {
                "file_name": staged_name,
                "source_path": str(source),
                "speaker": speaker,
                "text": text,
                "split": row.get("split", "train"),
                "materialization": materialization,
            }
        )

    (STAGE_DATASET / "metadata.csv").write_text(
        "\n".join(metadata_lines) + "\n", encoding="utf-8"
    )
    (WORK_ROOT / "source_manifest.json").write_text(
        json.dumps(
            {
                "source_metadata": str(metadata_path),
                "source_audio_root": str(audio_root),
                "count": len(manifest),
                "rows": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"staged {len(manifest)} WAVs: {STAGE_DATASET}", flush=True)
    print(f"metadata: {STAGE_DATASET / 'metadata.csv'}", flush=True)
    return STAGE_DATASET


def official_runtime() -> Path:
    local_finetune = SOURCE_ROOT / "finetune"
    local_train = local_finetune / "train_lora.py"
    local_prepare = local_finetune / "prepare_dataset.py"
    if local_train.is_file() and local_prepare.is_file():
        marker = local_train.read_text(encoding="utf-8", errors="ignore")
        if "load_v3_turbo_checkpoint" in marker and "--target" in marker:
            return local_finetune

    runtime_root = WORK_ROOT / "_official_runtime"
    for relative, filename in OFFICIAL_FILES.items():
        destination = runtime_root / "finetune" / relative
        if destination.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = OFFICIAL_BASE_URL + filename
        print(f"fetch official v3 fine-tune file: {url}", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                destination.write_bytes(response.read())
        except Exception as exc:
            raise RuntimeError(
                f"Cannot fetch official fine-tune source {url}: {exc}"
            ) from exc
    return runtime_root / "finetune"


def run_official(script: Path, arguments: list[str], runtime_finetune: Path) -> None:
    env = os.environ.copy()
    source_src = SOURCE_ROOT / "src"
    python_paths = [str(source_src), str(runtime_finetune)]
    old_pythonpath = env.get("PYTHONPATH")
    if old_pythonpath:
        python_paths.append(old_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    command = [sys.executable, "-u", str(script), *arguments]
    print("run:", " ".join(command), flush=True)
    # The official v3 helpers intentionally reject paths outside their current
    # working directory.  Run them from the repository root so staged data and
    # outputs under train/process remain inside the allowed workspace.
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def prepare(args: argparse.Namespace) -> None:
    runtime_finetune = official_runtime()
    dataset_dir = prepare_stage(args)
    output_parquet = dataset_dir / "train.parquet"
    run_official(
        runtime_finetune / "prepare_dataset.py",
        [
            "--dataset-dir",
            str(dataset_dir),
            "--out",
            str(output_parquet),
            "--base",
            BASE_MODEL,
            "--speaker",
            "nghean_30",
            "--min-sec",
            "1",
            "--max-sec",
            str(args.max_sec),
        ],
        runtime_finetune,
    )
    if not output_parquet.is_file():
        raise RuntimeError(f"Official preparation did not create {output_parquet}")
    print(f"PREPARE COMPLETE: {output_parquet}", flush=True)


def train(args: argparse.Namespace) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "Official VieNeu-TTS v3 LoRA training requires CUDA. "
            "Run --prepare on CPU, then run --train on a CUDA Windows/Ubuntu machine."
        )
    parquet = STAGE_DATASET / "train.parquet"
    if not parquet.is_file():
        raise FileNotFoundError(
            f"Missing {parquet}. Run --prepare first (or --all)."
        )
    runtime_finetune = official_runtime()
    output_dir = TRAINING_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "--data",
        str(parquet),
        "--run",
        RUN_NAME,
        "--output-dir",
        str(output_dir),
        "--base",
        BASE_MODEL,
        "--subfolder",
        "update",
        "--r",
        str(args.r),
        "--alpha",
        str(args.alpha),
        "--dropout",
        str(args.dropout),
        "--target",
        args.target,
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--lr",
        str(args.lr),
        "--max-steps",
        str(args.max_steps),
        "--eval-every",
        str(args.eval_every),
        "--save-every",
        str(args.save_every),
        "--log-every",
        str(args.log_every),
        "--max-length",
        str(args.max_length),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
    ]
    if not args.no_grad_checkpoint:
        command.append("--grad-checkpoint")
    if not args.no_merge:
        command.append("--merge")
    run_official(runtime_finetune / "train_lora.py", command, runtime_finetune)
    result_root = output_dir / RUN_NAME
    print(f"TRAIN COMPLETE: {result_root}", flush=True)
    print(f"adapter: {result_root / 'adapter'}", flush=True)
    if not args.no_merge:
        print(f"merged: {result_root / 'merged'}", flush=True)


def main() -> None:
    configure_console()
    args = parse_args()
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    if args.prepare or args.all:
        prepare(args)
    if args.train or args.all:
        train(args)


if __name__ == "__main__":
    main()
