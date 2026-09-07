"""Advanced Nghệ An LoRA training pipeline for VieNeu-TTS v2.

Goals
-----
- Stay close to the official VieNeu-TTS finetune flow:
    metadata -> official filter_data.py -> official encode_data.py
    -> official VieNeuDataset/LoRA -> train -> merge -> inference
- Keep the useful protections discovered during the pilot work:
    * max_len=2048 (official default)
    * reject only samples that would lose SPEECH_GENERATION_END
    * mask right-padding from labels
    * speaker-disjoint validation when enough speakers exist
    * conservative/adaptive LoRA schedule
    * no pre-training baseline gate
    * post-training base/merged A-B WAVs + JSON report
- Work on Linux/Ubuntu and Windows paths.
- Optionally train all voices, male only, or female only.

This script does NOT modify the original VieNeu source, v3 pipeline, or source audio.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# CPU / environment
# ---------------------------------------------------------------------------

CPU_THREADS = max(1, os.cpu_count() or 1)
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "source_code" / "audio_model"
SOURCE_FINETUNE = SOURCE_ROOT / "finetune"

sys.path.insert(0, str(SOURCE_ROOT / "src"))
sys.path.insert(0, str(SOURCE_ROOT))

PREPARED = ROOT / "train" / "output" / "nghean_test2_accent_scale"
MANIFEST = PREPARED / "nghean_eligible_audio_manifest.csv"
SOURCE_METADATA = ROOT / "train" / "metadata_na_candidates.csv"

BASE_MODEL = "pnnbao-ump/VieNeu-TTS-0.3B"
MAX_LEN = 2048
SEED = 37

TEST_TEXTS = [
    "Hôm nay tôi đang thử nghiệm giọng nói tiếng Việt.",
    "Tôi xin chào quý vị và các bạn.",
    "Chương trình hôm nay có nhiều thông tin đáng chú ý.",
]

# Diagnostic inference only. These values never gate training.
INFER_TEMPERATURE = 0.35
INFER_TOP_K = 25


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Advanced Nghệ An VieNeu-TTS v2 LoRA training pipeline."
    )
    p.add_argument(
        "--run-name",
        default="nghean_v2_advanced",
        help="Output directory name under train/output/.",
    )
    p.add_argument(
        "--gender",
        choices=("all", "male", "female"),
        default="all",
        help="Optional speaker gender filter when gender metadata is available.",
    )
    p.add_argument(
        "--train-speakers",
        type=int,
        default=0,
        help="0 = use every eligible train speaker; otherwise keep the N speakers with most samples.",
    )
    p.add_argument(
        "--max-per-speaker",
        type=int,
        default=0,
        help="0 = no cap; otherwise cap samples per speaker to reduce speaker dominance.",
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--epochs",
        type=float,
        default=0.0,
        help="0 = adaptive schedule; otherwise explicit epoch count.",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=0.0,
        help="0 = adaptive schedule; otherwise explicit learning rate.",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="-1 = auto (up to 16 workers), 0 = main process only.",
    )
    p.add_argument(
        "--validation-ratio",
        type=float,
        default=0.10,
        help="Speaker-disjoint validation ratio. Set 0 to disable validation.",
    )
    p.add_argument(
        "--skip-infer",
        action="store_true",
        help="Train/merge only; do not create post-training A-B WAV files.",
    )
    p.add_argument(
        "--skip-merge",
        action="store_true",
        help="Keep LoRA adapter only.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing run directory with the same name before starting.",
    )
    p.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Resume Trainer from an existing adapter/checkpoint directory.",
    )
    p.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable BF16 when a CUDA/cuBLAS driver is unstable; use FP16 instead.",
    )
    p.add_argument(
        "--fp32",
        action="store_true",
        help="Use full FP32 for maximum CUDA stability; slower and uses more VRAM.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_gender(value: str) -> str | None:
    v = (value or "").strip().lower()
    if not v:
        return None
    male = {"male", "m", "man", "nam", "男"}
    female = {"female", "f", "woman", "nữ", "nu", "女"}
    if v in male:
        return "male"
    if v in female:
        return "female"
    if "nam" == v:
        return "male"
    if "nữ" in v or "female" in v:
        return "female"
    if "male" in v:
        return "male"
    return None


def resolve_audio_path(raw_value: str) -> Path | None:
    """Resolve Linux/Windows/relative paths against the current repository."""
    raw = (raw_value or "").strip()
    if not raw:
        return None

    candidates: list[Path] = []
    p = Path(raw)
    candidates.append(p)

    if not p.is_absolute():
        candidates.append(ROOT / "train" / p)
        candidates.append(ROOT / p)

    normalized = raw.replace("\\", "/")
    lower = normalized.lower()

    # Re-map an absolute path copied from another OS/machine if it contains
    # the repository's train/ subtree.
    marker = "/train/"
    idx = lower.find(marker)
    if idx >= 0:
        rel = normalized[idx + len(marker):]
        candidates.append(ROOT / "train" / Path(rel))

    marker2 = "train/"
    if lower.startswith(marker2):
        candidates.append(ROOT / Path(normalized))

    # Deduplicate candidates while preserving order.
    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        try:
            if c.is_file():
                return c.resolve()
        except OSError:
            pass
    return None


def read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def qc_is_eligible(row: dict[str, str]) -> bool:
    status = (row.get("status") or "clean").strip().lower()
    qc = (row.get("qc_status") or "pass").strip().lower()
    decision = (row.get("decision") or "").strip().lower()

    bad_status = {"reject", "rejected", "bad", "drop", "remove", "loại", "loai"}
    bad_decision = {"reject", "rejected", "bad", "drop", "remove", "loại", "loai"}
    good_qc = {"pass", "clean", "acceptable", "good", "ok", ""}

    if status in bad_status or decision in bad_decision:
        return False
    if qc not in good_qc:
        return False
    return True


def load_source_rows(gender: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Combine manifest + source metadata, normalize paths, and keep clean rows."""
    source_rows = read_csv_if_exists(MANIFEST) + read_csv_if_exists(SOURCE_METADATA)
    if not source_rows:
        raise RuntimeError(
            f"No source metadata found. Expected {MANIFEST} and/or {SOURCE_METADATA}"
        )

    clean: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    gender_seen = Counter()

    for row in source_rows:
        split = (row.get("split") or "train").strip().lower()
        if split not in {"train", "valid", "test"}:
            continue
        if not qc_is_eligible(row):
            continue

        text = (row.get("transcript") or row.get("text") or "").strip()
        if not text:
            continue

        path = resolve_audio_path(row.get("local_path") or row.get("downloaded_file") or "")
        if path is None:
            continue

        try:
            duration = float(
                row.get("duration_sec")
                or row.get("duration")
                or row.get("duration_metadata")
                or 0
            )
        except (TypeError, ValueError):
            duration = 0.0

        # Official README recommends roughly 3-15 seconds.
        if not (3.0 <= duration <= 15.0):
            continue

        speaker = (row.get("speakerID") or row.get("speaker_id") or "").strip()
        if not speaker:
            speaker = f"speaker_{len(clean):05d}"

        g = normalize_gender(
            row.get("gender")
            or row.get("sex")
            or row.get("speaker_gender")
            or ""
        )
        if g:
            gender_seen[g] += 1
        if gender != "all" and g is not None and g != gender:
            continue
        # If gender metadata is absent, do not silently throw the row away.
        # The report will state that the requested gender could not be enforced.
        if gender != "all" and g is None:
            continue

        key = str(path)
        if key in seen_paths:
            continue
        seen_paths.add(key)

        clean.append(
            {
                "split": split,
                "speakerID": speaker,
                "gender": g,
                "local_path": str(path),
                "filename": path.name,
                "transcript": text,
                "duration_sec": duration,
            }
        )

    meta = {
        "source_rows_seen": len(source_rows),
        "eligible_rows": len(clean),
        "gender_counts_detected": dict(gender_seen),
        "gender_filter": gender,
    }

    if gender != "all" and not clean:
        raise RuntimeError(
            f"Gender filter '{gender}' produced zero rows. "
            "The source metadata may not contain usable gender labels."
        )
    return clean, meta


def select_training_rows(
    rows: list[dict[str, Any]],
    train_speakers: int,
    max_per_speaker: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Use train split for training; keep test/valid rows as reference candidates."""
    train_rows = [r for r in rows if r["split"] == "train"]
    reference_rows = [r for r in rows if r["split"] in {"test", "valid"}]

    if not train_rows:
        raise RuntimeError("No eligible rows in the train split.")

    counts = Counter(r["speakerID"] for r in train_rows)
    speakers = [
        s for s, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    ]
    if train_speakers > 0:
        speakers = speakers[:train_speakers]
    speaker_set = set(speakers)
    train_rows = [r for r in train_rows if r["speakerID"] in speaker_set]

    # Optional deterministic cap. This prevents one speaker from dominating an
    # accent adapter when a future dataset becomes unbalanced.
    if max_per_speaker > 0:
        rng = random.Random(seed)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in train_rows:
            grouped[row["speakerID"]].append(row)
        capped: list[dict[str, Any]] = []
        for speaker in sorted(grouped):
            group = grouped[speaker]
            rng.shuffle(group)
            capped.extend(group[:max_per_speaker])
        train_rows = capped

    train_rows.sort(
        key=lambda r: (
            0 if 4.0 <= r["duration_sec"] <= 10.0 else 1,
            r["speakerID"],
            r["filename"],
        )
    )

    stats = {
        "train_rows_selected": len(train_rows),
        "train_speakers_selected": len({r["speakerID"] for r in train_rows}),
        "reference_rows_available": len(reference_rows),
        "duration_sec_total": round(sum(r["duration_sec"] for r in train_rows), 3),
    }
    return train_rows, reference_rows, stats


# ---------------------------------------------------------------------------
# Official preprocessing
# ---------------------------------------------------------------------------

def prepare_run_dir(run: Path, overwrite: bool) -> None:
    if run.exists():
        if not overwrite:
            raise RuntimeError(
                f"{run} already exists. Use --overwrite or choose another --run-name."
            )
        shutil.rmtree(run)
    run.mkdir(parents=True, exist_ok=True)


def stage_dataset(train_rows: list[dict[str, Any]], dataset_dir: Path) -> dict[str, Any]:
    raw_dir = dataset_dir / "raw_audio"
    raw_dir.mkdir(parents=True, exist_ok=True)

    staged = []
    for i, row in enumerate(train_rows):
        safe_name = f"{row['speakerID']}__{i:05d}__{Path(row['filename']).name}"
        dst = raw_dir / safe_name
        shutil.copy2(row["local_path"], dst)
        staged.append(
            {
                "filename": safe_name,
                "speakerID": row["speakerID"],
                "text": row["transcript"],
                "duration_sec": row["duration_sec"],
            }
        )

    metadata = dataset_dir / "metadata.csv"
    with metadata.open("w", encoding="utf-8", newline="") as f:
        for r in staged:
            f.write(f"{r['filename']}|{r['text']}\n")

    return {
        "staged": staged,
        "metadata": metadata,
        "raw_dir": raw_dir,
    }


def official_filter_and_encode(dataset_dir: Path, max_samples: int) -> Path:
    """Run the repository's official filter and NeuCodec encoder."""
    # Apply the machine-wide CPU configuration before the official encoder
    # imports torch/librosa.  Audio loading is parallelized by encode_data.py;
    # NeuCodec itself remains a single GPU model to avoid VRAM duplication.
    import torch

    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(max(1, min(4, CPU_THREADS)))
    os.environ.setdefault("VIENEU_ENCODE_WORKERS", str(CPU_THREADS))
    from finetune.data_scripts.filter_data import filter_and_process_dataset
    from finetune.data_scripts.encode_data import encode_dataset

    filter_and_process_dataset(dataset_dir=str(dataset_dir))

    cleaned = dataset_dir / "metadata_cleaned.csv"
    if not cleaned.is_file():
        raise RuntimeError("Official filter_data.py did not create metadata_cleaned.csv")

    encode_dataset(dataset_dir=str(dataset_dir), max_samples=max_samples)

    encoded = dataset_dir / "metadata_encoded.csv"
    if not encoded.is_file():
        raise RuntimeError("Official encode_data.py did not create metadata_encoded.csv")
    return encoded


# ---------------------------------------------------------------------------
# Token audit / safe encoded set
# ---------------------------------------------------------------------------

def token_audit_and_filter(
    encoded: Path,
    tokenizer: Any,
    run: Path,
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    """Reject only invalid/context-overflow samples instead of aborting the run."""
    from vieneu_utils.phonemize_text import phonemize_with_dict

    start_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
    end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    with encoded.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|", 2)
            if len(parts) != 3:
                continue
            filename, text, codes_json = parts
            try:
                codes = json.loads(codes_json)
                phones = phonemize_with_dict(text)
            except Exception as exc:
                rejected.append(
                    {
                        "filename": filename,
                        "reason": f"parse_or_phonemize_error:{type(exc).__name__}",
                    }
                )
                continue

            codes_str = "".join(f"<|speech_{i}|>" for i in codes)
            chat = (
                f"<|TEXT_PROMPT_START|>{phones}<|TEXT_PROMPT_END|>"
                f"<|SPEECH_GENERATION_START|>{codes_str}<|SPEECH_GENERATION_END|>"
            )
            ids = tokenizer.encode(chat)
            start_pos = ids.index(start_id) if start_id in ids else -1
            end_pos = ids.index(end_id) if end_id in ids else -1
            valid = (
                start_pos >= 0
                and end_pos >= start_pos
                and len(ids) <= MAX_LEN
            )

            speaker = filename.split("__", 1)[0]
            item = {
                "filename": filename,
                "speakerID": speaker,
                "text": text,
                "codes": codes,
                "speech_code_count": len(codes),
                "total_token_count": len(ids),
                "has_speech_start": start_pos >= 0,
                "has_speech_end": end_pos >= 0,
                "eos_in_labels": start_pos >= 0 and end_pos >= start_pos,
                "over_context": len(ids) > MAX_LEN,
                "reason": "" if valid else "invalid_or_context_overflow",
            }
            (accepted if valid else rejected).append(item)

    audit_csv = run / "dataset_token_audit.csv"
    fields = [
        "filename",
        "speakerID",
        "speech_code_count",
        "total_token_count",
        "has_speech_start",
        "has_speech_end",
        "eos_in_labels",
        "over_context",
        "reason",
    ]
    with audit_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in accepted + rejected:
            w.writerow({k: r.get(k, "") for k in fields})

    safe_encoded = encoded.parent / "metadata_encoded_safe.csv"
    with safe_encoded.open("w", encoding="utf-8") as f:
        for r in accepted:
            f.write(
                f"{r['filename']}|{r['text']}|"
                f"{json.dumps(r['codes'], separators=(',', ':'))}\n"
            )

    if not accepted:
        raise RuntimeError("Token audit left zero trainable samples.")

    return safe_encoded, accepted, rejected


# ---------------------------------------------------------------------------
# Speaker-disjoint validation
# ---------------------------------------------------------------------------

def split_encoded_by_speaker(
    accepted: list[dict[str, Any]],
    dataset_dir: Path,
    seed: int,
    validation_ratio: float,
) -> tuple[Path, Path | None, list[dict[str, Any]], list[dict[str, Any]]]:
    speakers = sorted({r["speakerID"] for r in accepted})
    rng = random.Random(seed)
    rng.shuffle(speakers)

    valid_speakers: set[str] = set()
    if validation_ratio > 0 and len(speakers) >= 5:
        n_valid = max(1, int(round(len(speakers) * validation_ratio)))
        n_valid = min(n_valid, max(1, len(speakers) - 2))
        valid_speakers = set(speakers[:n_valid])

    train_rows = [r for r in accepted if r["speakerID"] not in valid_speakers]
    valid_rows = [r for r in accepted if r["speakerID"] in valid_speakers]

    train_path = dataset_dir / "train_encoded.csv"
    valid_path = dataset_dir / "valid_encoded.csv" if valid_rows else None

    def write(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(
                    f"{r['filename']}|{r['text']}|"
                    f"{json.dumps(r['codes'], separators=(',', ':'))}\n"
                )

    write(train_path, train_rows)
    if valid_path is not None:
        write(valid_path, valid_rows)

    if not train_rows:
        raise RuntimeError("Speaker split left zero training samples.")
    return train_path, valid_path, train_rows, valid_rows


# ---------------------------------------------------------------------------
# Adaptive training
# ---------------------------------------------------------------------------

def adaptive_hparams(
    n_train: int,
    epochs_arg: float,
    lr_arg: float,
) -> tuple[float, float]:
    """Conservative accent adaptation schedule that preserves the base model."""
    if epochs_arg > 0:
        epochs = epochs_arg
    elif n_train < 50:
        epochs = 1.0
    elif n_train < 150:
        epochs = 1.5
    elif n_train < 500:
        epochs = 2.0
    else:
        epochs = 2.5

    if lr_arg > 0:
        lr = lr_arg
    elif n_train < 150:
        lr = 1e-5
    elif n_train < 500:
        lr = 1.5e-5
    else:
        lr = 2e-5
    return epochs, lr


def train_lora(
    train_path: Path,
    valid_path: Path | None,
    adapter_dir: Path,
    seed: int,
    batch_size: int,
    grad_accum: int,
    workers_arg: int,
    epochs_arg: float,
    lr_arg: float,
    resume_from_checkpoint: Path | None = None,
    no_bf16: bool = False,
    fp32: bool = False,
) -> dict[str, Any]:
    import torch
    from peft import get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )
    from finetune.train import VieNeuDataset
    from finetune.configs.lora_config import lora_config

    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(max(1, min(4, CPU_THREADS)))
    except RuntimeError:
        pass

    if torch.cuda.is_available():
        # The RTX 3080 + this Qwen3/LoRA graph is unstable with TF32 CUBLAS
        # kernels (CUBLAS_STATUS_EXECUTION_FAILED during the MLP).  Keep all
        # tensors on CUDA, but use deterministic FP32 GEMMs for this path.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = bool(
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        and not no_bf16
        and not fp32
    )
    dtype = (
        torch.float32
        if fp32 or not torch.cuda.is_available()
        else torch.bfloat16
        if use_bf16
        else torch.float16
    )

    # Keep training on exactly one device.  Do not use ``device_map`` here:
    # Accelerate treats even a one-entry map as a model-parallel placement and
    # Trainer then skips its normal device move.  The 0.3B model fits on one
    # GPU, so load it normally and move the complete PEFT model explicitly.
    train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        dtype=dtype,
    )
    model = get_peft_model(model, lora_config)
    model = model.to(train_device)
    model.config.use_cache = False
    model.print_trainable_parameters()

    class SafeVieNeuDataset(VieNeuDataset):
        """Official preprocessing, but right-padding never contributes to loss."""

        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            item["labels"] = item["labels"].masked_fill(
                item["attention_mask"] == 0, -100
            )
            return item

    train_ds = SafeVieNeuDataset(str(train_path), tokenizer, max_len=MAX_LEN)
    valid_ds = (
        SafeVieNeuDataset(str(valid_path), tokenizer, max_len=MAX_LEN)
        if valid_path is not None
        else None
    )

    epochs, lr = adaptive_hparams(len(train_ds), epochs_arg, lr_arg)

    if workers_arg < 0:
        workers = min(CPU_THREADS, 16)
    else:
        workers = max(0, min(CPU_THREADS, workers_arg))

    steps_per_epoch = max(
        1, math.ceil(len(train_ds) / max(1, batch_size * grad_accum))
    )
    total_steps_est = max(1, int(math.ceil(steps_per_epoch * epochs)))
    logging_steps = max(1, total_steps_est // 10)
    eval_steps = max(1, total_steps_est // 4)
    save_steps = eval_steps

    has_eval = valid_ds is not None and len(valid_ds) > 0

    training_kwargs = dict(
        output_dir=str(adapter_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        logging_steps=logging_steps,
        report_to="none",
        dataloader_num_workers=workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_persistent_workers=workers > 0,
        remove_unused_columns=False,
        bf16=use_bf16,
        fp16=bool(torch.cuda.is_available() and not use_bf16 and not fp32),
        seed=seed,
        data_seed=seed,
        save_total_limit=3,
    )

    callbacks = []
    if has_eval:
        training_kwargs.update(
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
    else:
        training_kwargs.update(
            eval_strategy="no",
            save_strategy="epoch",
            load_best_model_at_end=False,
        )

    args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )

    t0 = time.time()
    resume_path = None
    if resume_from_checkpoint is not None:
        resume_path = (
            resume_from_checkpoint
            if resume_from_checkpoint.is_absolute()
            else ROOT / resume_from_checkpoint
        )
        if not resume_path.is_dir():
            raise FileNotFoundError(f"Checkpoint không tồn tại: {resume_path}")
        print(f"🦜 Resume từ checkpoint: {resume_path}")

    result = trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
    runtime_wall = time.time() - t0

    # When load_best_model_at_end=True, this saves the best weights currently
    # loaded by Trainer into the stable adapter root.
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)

    metrics = dict(result.metrics)
    metrics.update(
        {
            "wall_runtime_sec": runtime_wall,
            "train_samples": len(train_ds),
            "valid_samples": len(valid_ds) if valid_ds is not None else 0,
            "train_speakers": len(
                {
                    line.split("__", 1)[0]
                    for line in train_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                }
            ),
            "epochs_requested": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "estimated_optimizer_steps": total_steps_est,
            "workers": workers,
            "cpu_threads": CPU_THREADS,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "bf16": use_bf16,
            "fp32": fp32,
            "device_map": "none; explicit single-device placement",
            "resume_from_checkpoint": str(resume_path) if resume_path else None,
        }
    )
    return metrics


# ---------------------------------------------------------------------------
# Merge + post-training A/B
# ---------------------------------------------------------------------------

def merge_adapter(adapter_dir: Path, merged_dir: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        dtype=dtype,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True
    ).save_pretrained(merged_dir)


def choose_reference(
    reference_rows: list[dict[str, Any]],
    fallback_train_rows: list[dict[str, Any]],
    train_speakers: set[str],
) -> dict[str, Any]:
    candidates = [
        r for r in reference_rows
        if r["speakerID"] not in train_speakers
    ]
    if not candidates:
        candidates = list(reference_rows)
    if not candidates:
        candidates = list(fallback_train_rows)
    if not candidates:
        raise RuntimeError("No reference audio is available for post-training inference.")

    candidates.sort(
        key=lambda r: (
            0 if 4.0 <= r["duration_sec"] <= 8.0 else 1,
            abs(r["duration_sec"] - 6.0),
            r["speakerID"],
            r["filename"],
        )
    )
    return candidates[0]


def infer_one(
    engine: Any,
    text: str,
    ref: dict[str, Any],
    output: Path,
    seed: int,
) -> float:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    audio = engine.infer(
        text,
        ref_audio=str(ref["local_path"]),
        ref_text=ref["transcript"],
        max_chars=256,
        temperature=INFER_TEMPERATURE,
        top_k=INFER_TOP_K,
        apply_watermark=False,
    )
    engine.save(audio, str(output))
    return float(len(audio) / engine.sample_rate)


def post_training_ab(
    merged_dir: Path,
    run: Path,
    reference: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    import torch
    from vieneu import Vieneu

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = run / "ab_test"
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for label, repo in (("base", BASE_MODEL), ("merged", str(merged_dir))):
        engine = Vieneu(
            mode="standard",
            backbone_repo=repo,
            backbone_device=device,
            codec_repo="neuphonic/neucodec",
            codec_device=device,
            gguf_filename=None,
        )
        try:
            for i, text in enumerate(TEST_TEXTS, 1):
                wav = out / f"{label}_{i:02d}.wav"
                duration = infer_one(engine, text, reference, wav, seed)
                results.append(
                    {
                        "model": label,
                        "index": i,
                        "text": text,
                        "wav": str(wav),
                        "duration_sec": round(duration, 3),
                        "runaway_warning": bool(duration > 15.0),
                    }
                )
        finally:
            engine.close()
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    run = ROOT / "train" / "output" / args.run_name
    dataset_dir = run / "dataset"
    adapter_dir = run / "adapter"
    merged_dir = run / "merged_model"

    prepare_run_dir(run, args.overwrite)

    report: dict[str, Any] = {
        "status": "started",
        "base_model": BASE_MODEL,
        "run": str(run),
        "seed": args.seed,
        "max_len": MAX_LEN,
        "gender": args.gender,
        "notes": [
            "No baseline gate is used. Training is never blocked by stochastic base inference.",
            "Reference audio is used only after training for A-B listening tests.",
        ],
    }
    write_report(run / "training_report.json", report)

    rows, source_meta = load_source_rows(args.gender)
    train_rows, reference_rows, select_stats = select_training_rows(
        rows,
        train_speakers=args.train_speakers,
        max_per_speaker=args.max_per_speaker,
        seed=args.seed,
    )

    stage_info = stage_dataset(train_rows, dataset_dir)
    encoded = official_filter_and_encode(
        dataset_dir,
        max_samples=max(1, len(train_rows) + 10),
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    safe_encoded, accepted, rejected = token_audit_and_filter(
        encoded, tokenizer, run
    )

    train_path, valid_path, train_safe, valid_safe = split_encoded_by_speaker(
        accepted,
        dataset_dir,
        seed=args.seed,
        validation_ratio=args.validation_ratio,
    )

    report.update(
        {
            "source": source_meta,
            "selection": select_stats,
            "official_metadata": str(stage_info["metadata"]),
            "encoded": str(encoded),
            "safe_encoded": str(safe_encoded),
            "token_audit": {
                "accepted_samples": len(accepted),
                "rejected_samples": len(rejected),
                "max_token_count": max(r["total_token_count"] for r in accepted),
                "over_context_rejected": sum(
                    1 for r in rejected if r.get("over_context")
                ),
                "samples_with_eos": sum(
                    1 for r in accepted if r.get("eos_in_labels")
                ),
            },
            "split": {
                "train_samples": len(train_safe),
                "validation_samples": len(valid_safe),
                "train_speakers": len({r["speakerID"] for r in train_safe}),
                "validation_speakers": len({r["speakerID"] for r in valid_safe}),
            },
        }
    )
    write_report(run / "training_report.json", report)

    metrics = train_lora(
        train_path=train_path,
        valid_path=valid_path,
        adapter_dir=adapter_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        workers_arg=args.workers,
        epochs_arg=args.epochs,
        lr_arg=args.learning_rate,
        resume_from_checkpoint=args.resume_from_checkpoint,
        no_bf16=args.no_bf16,
        fp32=args.fp32,
    )
    report["training"] = metrics
    report["adapter"] = str(adapter_dir)
    write_report(run / "training_report.json", report)

    if not args.skip_merge:
        merge_adapter(adapter_dir, merged_dir)
        report["merged_model"] = str(merged_dir)

    if not args.skip_merge and not args.skip_infer:
        train_speaker_set = {r["speakerID"] for r in train_safe}
        reference = choose_reference(
            reference_rows,
            fallback_train_rows=train_rows,
            train_speakers=train_speaker_set,
        )
        report["reference"] = reference

        try:
            ab = post_training_ab(
                merged_dir=merged_dir,
                run=run,
                reference=reference,
                seed=args.seed,
            )
            report["ab_test"] = ab
            report["ab_runaway_count"] = sum(
                1 for r in ab if r["runaway_warning"]
            )
        except Exception as exc:
            # Inference diagnostics must not invalidate a completed LoRA train.
            report["ab_test_error"] = f"{type(exc).__name__}: {exc}"

    report["status"] = "completed"
    report["warnings"] = [
        "Listen to the generated WAV files before judging accent quality.",
        "If accent is weak but pronunciation is stable, increase clean regional data before aggressively increasing learning rate.",
        "For a male-only or female-only adapter, rerun with --gender male/female if the metadata contains gender labels.",
    ]
    write_report(run / "training_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
