"""Accent-focused VieNeu-TTS v3 Turbo LoRA runner for Nghá»‡ An.

This wrapper deliberately keeps the *official* VieNeu-TTS v3 Turbo pipeline:
    prepare_dataset.py -> train_lora.py -> optional official merge

What this wrapper changes is the experiment design around that official flow:
- select Nghá»‡ An rows from as few speakers as possible instead of spreading
  30 clips across many speakers;
- prefer clean 4-15 s clips first, then longer usable clips;
- verify real WAV duration before staging so official prepare does not silently
  throw away many selected clips;
- use a materially longer LoRA run than the old 30-step smoke test;
- use a stronger-but-still-conventional LoRA capacity (r32 / alpha64);
- keep target="backbone", i.e. the supported official v3 training target;
- keep adapter + merged output for A/B testing.

IMPORTANT
---------
This does not modify VieNeu-TTS source files. It only calls the official v3
fine-tuning scripts from source_code/audio_model/finetune.

The current VieNeu-TTS README recommends roughly 10-30 minutes of clean audio
for one voice. This script warns when the selected dataset is below 10 minutes.
For a regional accent, fewer consistent Nghá»‡ An speakers are preferable to
many unrelated speakers with only one utterance each.
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
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Console / paths
# ---------------------------------------------------------------------------

def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


HERE = Path(__file__).resolve().parent
# /.../vie_neu/train/process/v3/test_1 -> repo root = parents[3]
PROJECT_ROOT = HERE.parents[3]
SOURCE_ROOT = PROJECT_ROOT / "source_code" / "audio_model"

RUN_NAME = "nghean_v3_turbo_30_accent"
WORK_ROOT = HERE / "output" / RUN_NAME
STAGE_DATASET = WORK_ROOT / "dataset"
STAGE_AUDIO = STAGE_DATASET / "raw_audio"
TRAINING_ROOT = WORK_ROOT / "training"

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
SOURCE_METADATA = (
    SOURCE_METADATA_PRIMARY
    if SOURCE_METADATA_PRIMARY.is_file()
    else SOURCE_METADATA_FALLBACK
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)

    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--train", action="store_true")
    action.add_argument("--all", action="store_true")

    ap.add_argument("--source-metadata", type=Path, default=SOURCE_METADATA)
    ap.add_argument("--source-audio-root", type=Path, default=SOURCE_AUDIO_ROOT)

    # Keep 30 as the direct A/B pilot, but selection is now speaker-concentrated.
    ap.add_argument("--limit", type=int, default=30)

    ap.add_argument(
        "--min-sec",
        type=float,
        default=1.0,
        help="Reject clips shorter than this before official preparation.",
    )
    ap.add_argument(
        "--max-sec",
        type=float,
        default=30.0,
        help="Reject clips longer than this before official preparation.",
    )
    ap.add_argument(
        "--preferred-min-sec",
        type=float,
        default=4.0,
        help="Clips in the preferred range are selected first.",
    )
    ap.add_argument(
        "--preferred-max-sec",
        type=float,
        default=15.0,
        help="Clips in the preferred range are selected first.",
    )
    ap.add_argument(
        "--max-speakers",
        type=int,
        default=0,
        help=(
            "0 = use the minimum number of speakers needed to reach --limit. "
            "Set e.g. 5 to hard-limit speaker diversity; preparation fails if "
            "those speakers cannot supply enough clips."
        ),
    )
    ap.add_argument("--overwrite-stage", action="store_true")

    # Accent-focused LoRA preset. These are wrapper defaults, not a claim that
    # the upstream authors prescribe them for every dataset.
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--target", default="backbone")

    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)

    # Old smoke test was 30 steps (~3.75 epochs on 29 train rows).
    # 160 steps is ~20 epochs at effective batch 4: enough to test whether the
    # accent actually moves, while checkpoints every 20 steps make overfit easy
    # to detect by listening/eval.
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=160)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=2)

    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-grad-checkpoint", action="store_true")
    ap.add_argument("--no-merge", action="store_true")

    return ap.parse_args()


# ---------------------------------------------------------------------------
# Metadata / audio helpers
# ---------------------------------------------------------------------------

def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if not (row.get("text") or "").strip():
            for key in ("transcript", "transcription"):
                value = (row.get(key) or "").strip()
                if value:
                    row["text"] = value
                    break
    return rows


def find_source_audio(row: dict[str, str], audio_root: Path) -> Path:
    local_raw = (row.get("local_path") or "").strip()
    if local_raw:
        local = Path(local_raw).expanduser()
        if local.is_file():
            return local

    split = (row.get("split") or "train").strip()
    speaker = (row.get("speakerID") or row.get("speaker") or "").strip()
    filename = (row.get("filename") or row.get("file_name") or "").strip()

    candidates = []
    if filename:
        candidates.append(audio_root / split / speaker / filename)
        candidates.append(audio_root / speaker / filename)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    if filename:
        matches = list(audio_root.rglob(filename))
        for candidate in matches:
            if speaker and speaker not in candidate.parts:
                continue
            return candidate

    raise FileNotFoundError(
        f"Audio not found for speaker={speaker!r}, filename={filename!r} "
        f"under {audio_root}"
    )


def metadata_duration(row: dict[str, str]) -> float:
    for key in (
        "duration_sec",
        "duration",
        "duration_metadata",
        "audio_duration",
    ):
        raw = (row.get(key) or "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    return 0.0


def wav_duration(path: Path, fallback: float = 0.0) -> float:
    """Read WAV duration without torch/ffmpeg; fall back to metadata if needed."""
    try:
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            frames = wf.getnframes()
            if rate > 0 and frames > 0:
                return frames / float(rate)
    except (wave.Error, EOFError, OSError):
        pass
    return fallback


def quality_penalty(row: dict[str, str]) -> int:
    """Generic conservative ranking only; no dataset-specific score is invented."""
    joined = " ".join(
        str(row.get(k) or "").lower()
        for k in (
            "quality",
            "quality_status",
            "status",
            "audio_status",
            "note",
            "notes",
        )
    )
    if any(x in joined for x in ("clipping", "clip", "bad", "reject", "noise")):
        return 2
    if any(x in joined for x in ("pass", "clean", "good", "ok")):
        return 0
    return 1


def speaker_of(row: dict[str, str]) -> str:
    return (
        row.get("speakerID")
        or row.get("speaker")
        or row.get("speaker_id")
        or ""
    ).strip()


def filename_of(row: dict[str, str]) -> str:
    return (
        row.get("filename")
        or row.get("file_name")
        or row.get("audio")
        or ""
    ).strip()


# ---------------------------------------------------------------------------
# Accent-focused selection
# ---------------------------------------------------------------------------

def collect_usable_rows(
    metadata_path: Path,
    audio_root: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = read_csv(metadata_path)
    usable: list[dict[str, Any]] = []

    for row in rows:
        split = (row.get("split") or "train").strip().lower()
        if split != "train":
            continue

        text = (row.get("text") or "").strip()
        speaker = speaker_of(row)
        filename = filename_of(row)
        if not text or not speaker or not filename:
            continue

        try:
            audio = find_source_audio(row, audio_root)
        except FileNotFoundError:
            continue

        duration = wav_duration(audio, metadata_duration(row))
        if duration <= 0:
            continue
        if duration < args.min_sec or duration > args.max_sec:
            continue

        usable.append(
            {
                "row": row,
                "audio": audio,
                "duration": duration,
                "speaker": speaker,
                "filename": filename,
                "text": text,
                "quality_penalty": quality_penalty(row),
            }
        )

    return usable


def select_rows_accent_focused(
    metadata_path: Path,
    audio_root: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Minimize speaker count first, then pick the cleanest useful clips.

    The previous pilot spread 30 clips across many speakers. For a regional
    adapter with only minutes of data, this makes the shared LoRA update weak:
    speaker/reference conditioning can explain much of each utterance while the
    backbone remains close to the base model.

    Here we rank speakers by how much usable Nghá»‡ An material they contribute
    and consume the strongest speakers first. We do NOT falsify speaker labels
    or pair unrelated speakers as one identity.
    """
    usable = collect_usable_rows(metadata_path, audio_root, args)

    if len(usable) < args.limit:
        raise RuntimeError(
            f"Only {len(usable)} usable train WAVs remain inside "
            f"[{args.min_sec}, {args.max_sec}] sec; need {args.limit}."
        )

    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in usable:
        by_speaker[item["speaker"]].append(item)

    def preferred(item: dict[str, Any]) -> bool:
        return (
            args.preferred_min_sec
            <= item["duration"]
            <= args.preferred_max_sec
        )

    for speaker, items in by_speaker.items():
        items.sort(
            key=lambda x: (
                x["quality_penalty"],
                0 if preferred(x) else 1,
                abs(x["duration"] - 8.0),
                x["filename"],
            )
        )

    ranked_speakers = sorted(
        by_speaker,
        key=lambda spk: (
            -len(by_speaker[spk]),
            -sum(x["duration"] for x in by_speaker[spk]),
            spk,
        ),
    )

    if args.max_speakers > 0:
        ranked_speakers = ranked_speakers[: args.max_speakers]
        available = sum(len(by_speaker[s]) for s in ranked_speakers)
        if available < args.limit:
            summary = {
                s: len(by_speaker[s])
                for s in ranked_speakers
            }
            raise RuntimeError(
                f"--max-speakers={args.max_speakers} supplies only "
                f"{available}/{args.limit} clips. Counts={summary}. "
                "Increase --max-speakers or reduce --limit."
            )

    selected: list[dict[str, Any]] = []
    used_speakers: list[str] = []

    for speaker in ranked_speakers:
        if len(selected) >= args.limit:
            break
        used_speakers.append(speaker)
        remaining = args.limit - len(selected)
        selected.extend(by_speaker[speaker][:remaining])

    if len(selected) != args.limit:
        raise RuntimeError(
            f"Selection produced {len(selected)} rows; expected {args.limit}."
        )

    counts = Counter(x["speaker"] for x in selected)
    total_min = sum(x["duration"] for x in selected) / 60.0

    print(
        f"accent-focused selection: {len(selected)} WAVs | "
        f"{len(counts)} speakers | {total_min:.2f} min",
        flush=True,
    )
    print(
        "speaker counts: "
        + ", ".join(f"{s}={n}" for s, n in counts.most_common()),
        flush=True,
    )

    if total_min < 10.0:
        print(
            "WARNING: selected audio is below the current upstream guidance "
            "of about 10-30 minutes for one fine-tuned voice. "
            "Training can still run, but accent transfer may remain limited.",
            flush=True,
        )

    if len(counts) > 8:
        print(
            "WARNING: more than 8 speakers are still required to fill this "
            "pilot. Regional accent may be diluted. Prefer adding more clips "
            "from the strongest existing speakers rather than new speakers.",
            flush=True,
        )

    return selected


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

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

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Audio root not found: {audio_root}")

    if args.overwrite_stage and STAGE_DATASET.exists():
        shutil.rmtree(STAGE_DATASET)

    STAGE_AUDIO.mkdir(parents=True, exist_ok=True)

    selected = select_rows_accent_focused(
        metadata_path=metadata_path,
        audio_root=audio_root,
        args=args,
    )

    manifest: list[dict[str, Any]] = []
    metadata_lines: list[str] = []

    for index, item in enumerate(selected, start=1):
        text = item["text"]
        speaker = item["speaker"]
        source = item["audio"]

        if "|" in text:
            raise ValueError(
                f"Transcript contains '|', unsupported by official metadata: {source}"
            )

        staged_name = f"{index:03d}_{source.name}"
        staged_path = STAGE_AUDIO / staged_name
        materialization = link_or_copy(source, staged_path)

        # Keep the real speaker id. Do not collapse unrelated people into one
        # fake identity; that would create contradictory speaker conditioning.
        metadata_lines.append(f"{staged_name}|{text}|{speaker}")

        manifest.append(
            {
                "file_name": staged_name,
                "source_path": str(source),
                "speaker": speaker,
                "text": text,
                "duration_sec": round(float(item["duration"]), 4),
                "quality_penalty": int(item["quality_penalty"]),
                "materialization": materialization,
            }
        )

    (STAGE_DATASET / "metadata.csv").write_text(
        "\n".join(metadata_lines) + "\n",
        encoding="utf-8",
    )

    manifest_payload = {
        "source_metadata": str(metadata_path),
        "source_audio_root": str(audio_root),
        "count": len(manifest),
        "speaker_count": len({x["speaker"] for x in manifest}),
        "total_minutes": round(
            sum(x["duration_sec"] for x in manifest) / 60.0,
            3,
        ),
        "selection": {
            "strategy": "minimum_speaker_count_then_clean_preferred_duration",
            "limit": args.limit,
            "min_sec": args.min_sec,
            "max_sec": args.max_sec,
            "preferred_min_sec": args.preferred_min_sec,
            "preferred_max_sec": args.preferred_max_sec,
            "max_speakers": args.max_speakers,
        },
        "rows": manifest,
    }

    (WORK_ROOT / "source_manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"staged {len(manifest)} WAVs: {STAGE_DATASET}", flush=True)
    print(f"metadata: {STAGE_DATASET / 'metadata.csv'}", flush=True)
    return STAGE_DATASET


# ---------------------------------------------------------------------------
# Official v3 runtime discovery
# ---------------------------------------------------------------------------

def official_runtime() -> Path:
    local_finetune = SOURCE_ROOT / "finetune"
    local_train = local_finetune / "train_lora.py"
    local_prepare = local_finetune / "prepare_dataset.py"

    if local_train.is_file() and local_prepare.is_file():
        marker = local_train.read_text(
            encoding="utf-8",
            errors="ignore",
        )
        # These markers distinguish the new v3 Turbo trainer from the old v1/v2
        # finetune directory.
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


def run_official(
    script: Path,
    arguments: list[str],
    runtime_finetune: Path,
) -> None:
    env = os.environ.copy()

    source_src = SOURCE_ROOT / "src"
    python_paths = [str(source_src), str(runtime_finetune)]
    old_pythonpath = env.get("PYTHONPATH")
    if old_pythonpath:
        python_paths.append(old_pythonpath)

    env["PYTHONPATH"] = os.pathsep.join(python_paths)

    command = [sys.executable, "-u", str(script), *arguments]
    print("run:", " ".join(command), flush=True)

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


# ---------------------------------------------------------------------------
# Official prepare
# ---------------------------------------------------------------------------

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
            "nghean_region",
            "--min-sec",
            str(args.min_sec),
            "--max-sec",
            str(args.max_sec),
        ],
        runtime_finetune,
    )

    if not output_parquet.is_file():
        raise RuntimeError(
            f"Official preparation did not create {output_parquet}"
        )

    print(f"PREPARE COMPLETE: {output_parquet}", flush=True)


# ---------------------------------------------------------------------------
# Official train
# ---------------------------------------------------------------------------

def write_training_plan(args: argparse.Namespace) -> None:
    payload = {
        "run_name": RUN_NAME,
        "base_model": BASE_MODEL,
        "official_pipeline": True,
        "accent_preset": {
            "r": args.r,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "target": args.target,
            "learning_rate": args.lr,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "eval_every": args.eval_every,
            "save_every": args.save_every,
            "max_length": args.max_length,
            "gradient_checkpointing": not args.no_grad_checkpoint,
            "merge": not args.no_merge,
        },
        "note": (
            "r32/a64 + 1e-4 + 160 steps is an accent-focused experiment built "
            "on top of the official v3 trainer. The upstream README recommends "
            "roughly 10-30 minutes clean audio for one voice; data consistency "
            "still matters more than simply increasing steps."
        ),
    }

    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    (TRAINING_ROOT / "training_plan.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train(args: argparse.Namespace) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "Official VieNeu-TTS v3 LoRA training requires CUDA. "
            "Run --prepare on CPU if needed, then --train on CUDA."
        )

    parquet = STAGE_DATASET / "train.parquet"
    if not parquet.is_file():
        raise FileNotFoundError(
            f"Missing {parquet}. Run --prepare first (or use --all)."
        )

    if args.r <= 0 or args.alpha <= 0:
        raise ValueError("--r and --alpha must be > 0")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.batch_size <= 0 or args.grad_accum <= 0:
        raise ValueError("--batch-size and --grad-accum must be > 0")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")

    runtime_finetune = official_runtime()
    output_dir = TRAINING_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    write_training_plan(args)

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

    print(
        "\nACCENT TRAIN PRESET\n"
        f"  LoRA          : r={args.r}, alpha={args.alpha}, dropout={args.dropout}\n"
        f"  target        : {args.target}\n"
        f"  LR            : {args.lr}\n"
        f"  steps         : {args.max_steps}\n"
        f"  effective batch: {args.batch_size * args.grad_accum}\n"
        f"  checkpoint    : every {args.save_every} steps\n",
        flush=True,
    )

    run_official(
        runtime_finetune / "train_lora.py",
        command,
        runtime_finetune,
    )

    result_root = output_dir / RUN_NAME
    print(f"TRAIN COMPLETE: {result_root}", flush=True)
    print(f"adapter: {result_root / 'adapter'}", flush=True)

    if not args.no_merge:
        print(f"merged: {result_root / 'merged'}", flush=True)

    print(
        "\nFor a strong-accent pilot, listen to intermediate checkpoints too. "
        "The checkpoint with the lowest eval loss is not automatically the one "
        "with the strongest regional accent.",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    configure_console()
    args = parse_args()

    if args.limit <= 1:
        raise ValueError("--limit must be > 1")
    if args.min_sec <= 0 or args.max_sec <= args.min_sec:
        raise ValueError("Invalid --min-sec/--max-sec")
    if (
        args.preferred_min_sec < args.min_sec
        or args.preferred_max_sec > args.max_sec
        or args.preferred_max_sec <= args.preferred_min_sec
    ):
        raise ValueError(
            "Preferred duration range must lie inside min/max duration."
        )
    if args.max_speakers < 0:
        raise ValueError("--max-speakers must be >= 0")

    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    if args.prepare or args.all:
        prepare(args)

    if args.train or args.all:
        train(args)


if __name__ == "__main__":
    main()


