"""Train exactly 30 safe Nghệ An utterances as a runtime PEFT LoRA.

Design
------
- Reuse the verified v2 preprocessing helpers from train_nghean_v2_advanced.py.
- Stage ALL eligible train candidates, run official filter + NeuCodec encode,
  audit token/context/EOS safety, THEN choose exactly 30 safe train rows.
- Keep the base model untouched. Save only the PEFT adapter.
- Up-weight only <|SPEECH_GENERATION_END|> targets in the causal token loss.
- Preserve the fast single-GPU / multi-CPU behavior used by the current v2
  training pipeline.

This file does not modify source_code/audio_model or the production v2 script.
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
# Environment / paths
# ---------------------------------------------------------------------------

CPU_THREADS = max(1, os.cpu_count() or 1)

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, str(CPU_THREADS))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("VIENEU_ENCODE_WORKERS", str(CPU_THREADS))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[4]
V2_DIR = ROOT / "train" / "process" / "v2"
SOURCE_ROOT = ROOT / "source_code" / "audio_model"

sys.path.insert(0, str(V2_DIR))
sys.path.insert(0, str(SOURCE_ROOT / "src"))
sys.path.insert(0, str(SOURCE_ROOT))

from train_nghean_v2_advanced import (  # noqa: E402
    BASE_MODEL,
    MAX_LEN,
    load_source_rows,
    official_filter_and_encode,
    prepare_run_dir,
    stage_dataset,
    token_audit_and_filter,
)

CODEC_MODEL = "neuphonic/neucodec"
DEFAULT_RUN_NAME = "nghean_v2_lora30_eos"
TARGET_SAFE_TRAIN = 30
DEFAULT_EOS_LOSS_WEIGHT = 5.0


# ---------------------------------------------------------------------------
# CLI / simple IO
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train exactly 30 safe Nghệ An rows as an adapter-only VieNeu v2 LoRA."
    )
    p.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    p.add_argument("--epochs", type=float, default=80.0)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--eos-loss-weight", type=float, default=DEFAULT_EOS_LOSS_WEIGHT)
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 = auto by VRAM: >=11GB -> 4, >=8GB -> 2, otherwise 1.",
    )
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="-1 = auto min(CPU threads, 16).",
    )
    p.add_argument("--seed", type=int, default=37)
    p.add_argument("--fast-gpu", action="store_true")
    p.add_argument("--fp32", action="store_true")
    p.add_argument(
        "--validation-limit",
        type=int,
        default=12,
        help="Maximum external valid/test utterances to encode; 0 = all.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Optional Trainer checkpoint directory.",
    )
    return p.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_pipe_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        for row in rows:
            f.write(
                f"{row['filename']}|{row['text']}|"
                f"{json.dumps(row['codes'], separators=(',', ':'))}\n"
            )


def bool_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


# ---------------------------------------------------------------------------
# Safe row selection
# ---------------------------------------------------------------------------

def select_exactly_30(
    accepted: list[dict[str, Any]],
    staged: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Choose exactly 30 already-safe encoded rows.

    Preference:
    1) speakers with multiple usable utterances;
    2) 5-8 second utterances;
    3) several passes over each speaker before globally filling the remainder.

    This deliberately avoids the old mistake of selecting 30 raw rows and
    allowing the official filter to shrink the actual training set afterward.
    """
    if len(accepted) < TARGET_SAFE_TRAIN:
        raise RuntimeError(
            f"Chỉ có {len(accepted)} safe encoded train samples sau filter/encode/audit; "
            f"cần đúng {TARGET_SAFE_TRAIN}. Không train."
        )

    duration_by_filename = {
        row["filename"]: float(row.get("duration_sec") or 0.0)
        for row in staged
    }

    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        by_speaker[row["speakerID"]].append(row)

    for speaker, rows in by_speaker.items():
        rows.sort(
            key=lambda row: (
                0
                if 5.0 <= duration_by_filename.get(row["filename"], 0.0) <= 8.0
                else 1,
                abs(duration_by_filename.get(row["filename"], 6.0) - 6.0),
                int(row.get("total_token_count", 0) or 0),
                row["filename"],
            )
        )

    speaker_counts = {
        speaker: len(rows)
        for speaker, rows in by_speaker.items()
    }
    speakers = sorted(
        by_speaker,
        key=lambda speaker: (-speaker_counts[speaker], speaker),
    )

    selected: list[dict[str, Any]] = []

    # Up to four utterances per speaker in round-robin order before filling.
    for round_index in range(4):
        for speaker in speakers:
            rows = by_speaker[speaker]
            if round_index < len(rows) and len(selected) < TARGET_SAFE_TRAIN:
                selected.append(rows[round_index])

    if len(selected) < TARGET_SAFE_TRAIN:
        already = {id(row) for row in selected}
        remaining = [
            row
            for row in accepted
            if id(row) not in already
        ]
        remaining.sort(
            key=lambda row: (
                -speaker_counts[row["speakerID"]],
                0
                if 5.0 <= duration_by_filename.get(row["filename"], 0.0) <= 8.0
                else 1,
                abs(duration_by_filename.get(row["filename"], 6.0) - 6.0),
                row["filename"],
            )
        )
        selected.extend(
            remaining[: TARGET_SAFE_TRAIN - len(selected)]
        )

    if len(selected) != TARGET_SAFE_TRAIN:
        raise RuntimeError(
            f"Internal selection error: selected={len(selected)}, "
            f"expected={TARGET_SAFE_TRAIN}."
        )

    return selected


def choose_external_validation_rows(
    source_rows: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in source_rows
        if row["split"] in {"valid", "test"}
    ]
    rows.sort(
        key=lambda row: (
            0 if 4.0 <= row["duration_sec"] <= 8.0 else 1,
            abs(row["duration_sec"] - 6.0),
            row["speakerID"],
            row["filename"],
        )
    )
    if limit > 0:
        rows = rows[:limit]
    return rows


def encode_external_validation(
    rows: list[dict[str, Any]],
    run: Path,
    tokenizer: Any,
) -> tuple[Path | None, list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return None, [], []

    valid_dir = run / "dataset" / "external_valid"
    valid_dir.mkdir(parents=True, exist_ok=True)

    stage_dataset(rows, valid_dir)
    encoded = official_filter_and_encode(
        valid_dir,
        max_samples=max(1, len(rows) + 10),
    )

    audit_dir = run / "valid_eos_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    safe_path, accepted, rejected = token_audit_and_filter(
        encoded,
        tokenizer,
        audit_dir,
    )

    if not accepted:
        return None, [], rejected

    target = run / "dataset" / "valid_encoded.csv"
    shutil.copy2(safe_path, target)
    return target, accepted, rejected


# ---------------------------------------------------------------------------
# EOS-weighted Trainer
# ---------------------------------------------------------------------------

class EOSWeightedTrainerMixin:
    """Causal LM loss with extra weight ONLY on speech-END target positions."""

    eos_loss_weight: float
    eos_token_id: int

    train_eos_targets: int
    train_normal_loss_sum: float
    train_eos_loss_sum: float
    train_normal_count: int
    train_eos_count: int

    def _init_eos_stats(self, weight: float, eos_token_id: int) -> None:
        if weight <= 0:
            raise ValueError("--eos-loss-weight must be > 0")

        self.eos_loss_weight = float(weight)
        self.eos_token_id = int(eos_token_id)

        self.train_eos_targets = 0
        self.train_normal_loss_sum = 0.0
        self.train_eos_loss_sum = 0.0
        self.train_normal_count = 0
        self.train_eos_count = 0

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        import torch
        import torch.nn.functional as F

        # Do not ask the backbone to compute its own unweighted loss; we only
        # need logits and calculate the exact weighted objective below.
        model_inputs = dict(inputs)
        labels = model_inputs.pop("labels")
        outputs = model(**model_inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)

        token_losses = F.cross_entropy(
            flat_logits,
            flat_labels,
            reduction="none",
            ignore_index=-100,
        )

        valid_mask = flat_labels.ne(-100)
        eos_mask = flat_labels.eq(self.eos_token_id) & valid_mask
        normal_mask = valid_mask & ~eos_mask

        valid_losses = token_losses[valid_mask]
        if valid_losses.numel() == 0:
            # Should never happen for a valid VieNeu item.
            loss = token_losses.sum() * 0.0
        else:
            weights = torch.ones_like(token_losses)
            weights[eos_mask] = self.eos_loss_weight
            valid_weights = weights[valid_mask]
            loss = (
                (valid_losses * valid_weights).sum()
                / valid_weights.sum().clamp_min(1.0)
            )

        # Only accumulate diagnostics during training, not validation passes.
        if model.training:
            with torch.no_grad():
                eos_count = int(eos_mask.sum().item())
                normal_count = int(normal_mask.sum().item())

                self.train_eos_targets += eos_count
                self.train_eos_count += eos_count
                self.train_normal_count += normal_count

                if normal_count:
                    self.train_normal_loss_sum += float(
                        token_losses[normal_mask]
                        .detach()
                        .float()
                        .sum()
                        .item()
                    )
                if eos_count:
                    self.train_eos_loss_sum += float(
                        token_losses[eos_mask]
                        .detach()
                        .float()
                        .sum()
                        .item()
                    )

        return (loss, outputs) if return_outputs else loss


def train_adapter(
    train_path: Path,
    valid_path: Path | None,
    adapter_dir: Path,
    args: argparse.Namespace,
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
    from finetune.configs.lora_config import lora_config
    from finetune.train import VieNeuDataset

    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(max(1, min(4, CPU_THREADS)))
    except RuntimeError:
        pass

    cuda = torch.cuda.is_available()
    if cuda:
        # Fast mode is intentionally the same philosophy as the working v2
        # pipeline: one explicit CUDA device, BF16/SDPA when available.
        torch.backends.cuda.matmul.allow_tf32 = bool(args.fast_gpu)
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    speech_end_id = tokenizer.convert_tokens_to_ids(
        "<|SPEECH_GENERATION_END|>"
    )
    if speech_end_id is None or int(speech_end_id) < 0:
        raise RuntimeError(
            "Base tokenizer does not expose <|SPEECH_GENERATION_END|>."
        )

    use_bf16 = bool(
        cuda
        and torch.cuda.is_bf16_supported()
        and not args.fp32
    )

    dtype = (
        torch.float32
        if args.fp32 or not cuda
        else torch.bfloat16
        if use_bf16
        else torch.float16
    )

    train_device = torch.device("cuda:0" if cuda else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="sdpa" if args.fast_gpu else "eager",
    )
    model = get_peft_model(model, lora_config)
    model = model.to(train_device)
    model.config.use_cache = False

    # On the user's ~9.8GB card, fast_gpu + batch=2 has already been proven to
    # fit, so do not forcibly re-enable checkpointing merely because VRAM <11GB.
    use_checkpointing = not args.fast_gpu
    if use_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    model.print_trainable_parameters()

    class SafeVieNeuDataset(VieNeuDataset):
        """Official dataset behavior, but right-padding never contributes loss."""

        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            item["labels"] = item["labels"].masked_fill(
                item["attention_mask"] == 0,
                -100,
            )
            return item

    train_ds = SafeVieNeuDataset(
        str(train_path),
        tokenizer,
        max_len=MAX_LEN,
    )
    valid_ds = (
        SafeVieNeuDataset(str(valid_path), tokenizer, max_len=MAX_LEN)
        if valid_path is not None and valid_path.is_file()
        else None
    )

    if len(train_ds) != TARGET_SAFE_TRAIN:
        raise RuntimeError(
            f"Trainer received {len(train_ds)} train samples, "
            f"expected exactly {TARGET_SAFE_TRAIN}."
        )

    if args.batch_size > 0:
        batch_size = args.batch_size
        vram_gb = None
    elif cuda:
        vram_gb = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024**3)
        )
        batch_size = 4 if vram_gb >= 11 else 2 if vram_gb >= 8 else 1
    else:
        vram_gb = None
        batch_size = 1

    workers = (
        min(CPU_THREADS, 16)
        if args.workers < 0
        else max(0, min(CPU_THREADS, args.workers))
    )

    steps_per_epoch = max(
        1,
        math.ceil(
            len(train_ds)
            / max(1, batch_size * args.grad_accum)
        ),
    )
    total_steps = max(
        1,
        int(math.ceil(steps_per_epoch * args.epochs)),
    )

    logging_steps = max(1, total_steps // 20)
    eval_steps = max(1, total_steps // 4)
    save_steps = eval_steps

    has_eval = valid_ds is not None and len(valid_ds) > 0

    training_kwargs: dict[str, Any] = dict(
        output_dir=str(adapter_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        logging_steps=logging_steps,
        report_to="none",
        dataloader_num_workers=workers,
        dataloader_pin_memory=cuda,
        dataloader_persistent_workers=workers > 0,
        remove_unused_columns=False,
        gradient_checkpointing=use_checkpointing,
        bf16=use_bf16,
        fp16=bool(cuda and not use_bf16 and not args.fp32),
        seed=args.seed,
        data_seed=args.seed,
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
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=3)
        )
    else:
        training_kwargs.update(
            eval_strategy="no",
            save_strategy="steps",
            save_steps=save_steps,
            load_best_model_at_end=False,
        )

    class EOSTrainer(EOSWeightedTrainerMixin, Trainer):
        pass

    training_args = TrainingArguments(**training_kwargs)

    trainer = EOSTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )
    trainer._init_eos_stats(
        args.eos_loss_weight,
        int(speech_end_id),
    )

    resume_path = None
    if args.resume_from_checkpoint is not None:
        resume_path = args.resume_from_checkpoint.expanduser()
        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path
        if not resume_path.is_dir():
            raise FileNotFoundError(
                f"Checkpoint does not exist: {resume_path}"
            )

    print(
        f"🦜 Train exactly {len(train_ds)} safe rows | "
        f"valid={len(valid_ds) if valid_ds is not None else 0} | "
        f"steps≈{total_steps} | batch={batch_size} | "
        f"grad_accum={args.grad_accum} | "
        f"eos_loss_weight={args.eos_loss_weight}",
        flush=True,
    )

    if vram_gb is not None:
        print(
            f"🦜 Auto batch theo VRAM {vram_gb:.1f} GB: "
            f"batch={batch_size}",
            flush=True,
        )

    t0 = time.time()
    result = trainer.train(
        resume_from_checkpoint=(
            str(resume_path)
            if resume_path is not None
            else None
        )
    )
    wall_runtime = time.time() - t0

    # If load_best_model_at_end=True, the Trainer has already restored the best
    # evaluated adapter weights before this stable save.
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    normal_token_loss = (
        trainer.train_normal_loss_sum
        / max(1, trainer.train_normal_count)
    )
    eos_token_loss = (
        trainer.train_eos_loss_sum
        / max(1, trainer.train_eos_count)
    )

    metrics = dict(result.metrics)
    metrics.update(
        {
            "wall_runtime_sec": wall_runtime,
            "train_samples": len(train_ds),
            "valid_samples": (
                len(valid_ds)
                if valid_ds is not None
                else 0
            ),
            "epochs_requested": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "estimated_optimizer_steps": total_steps,
            "workers": workers,
            "cpu_threads": CPU_THREADS,
            "bf16": use_bf16,
            "fp32": args.fp32,
            "fast_gpu": args.fast_gpu,
            "gradient_checkpointing": use_checkpointing,
            "eos_loss_weight": args.eos_loss_weight,
            "number_of_train_eos_targets_seen": (
                trainer.train_eos_targets
            ),
            "normal_token_loss": normal_token_loss,
            "eos_token_loss": eos_token_loss,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "adapter_only": True,
        }
    )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be > 0")
    if args.eos_loss_weight <= 0:
        raise ValueError("--eos-loss-weight must be > 0")
    if args.grad_accum <= 0:
        raise ValueError("--grad-accum must be > 0")

    random.seed(args.seed)

    run = ROOT / "train" / "output" / args.run_name
    dataset_dir = run / "dataset"
    adapter_dir = run / "adapter"

    prepare_run_dir(run, args.overwrite)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(
        "🦜 Tìm toàn bộ candidate Nghệ An; "
        "không cắt xuống 30 trước filter/encode...",
        flush=True,
    )

    source_rows, source_meta = load_source_rows("all")

    train_candidates = [
        row
        for row in source_rows
        if row["split"] == "train"
    ]
    external_validation = choose_external_validation_rows(
        source_rows,
        args.validation_limit,
    )

    if len(train_candidates) < TARGET_SAFE_TRAIN:
        raise RuntimeError(
            f"Nguồn chỉ có {len(train_candidates)} eligible train candidates; "
            f"cần ít nhất {TARGET_SAFE_TRAIN}."
        )

    # Prefer multi-utterance speakers and useful durations before staging. We
    # still stage ALL eligible candidates; ordering only affects deterministic
    # file naming / official max-sample traversal.
    speaker_counts = Counter(
        row["speakerID"]
        for row in train_candidates
    )
    train_candidates.sort(
        key=lambda row: (
            -speaker_counts[row["speakerID"]],
            0 if 5.0 <= row["duration_sec"] <= 8.0 else 1,
            abs(row["duration_sec"] - 6.0),
            row["speakerID"],
            row["filename"],
        )
    )

    staged_info = stage_dataset(
        train_candidates,
        dataset_dir,
    )

    print(
        f"🦜 Staged {len(train_candidates)} train candidates; "
        "chạy official filter + NeuCodec encode...",
        flush=True,
    )

    encoded = official_filter_and_encode(
        dataset_dir,
        max_samples=max(100000, len(train_candidates) + 10),
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _, accepted_all, rejected_all = token_audit_and_filter(
        encoded,
        tokenizer,
        run,
    )

    print(
        f"🦜 Safe sau filter/encode/token audit: "
        f"{len(accepted_all)} | rejected={len(rejected_all)}",
        flush=True,
    )

    selected = select_exactly_30(
        accepted_all,
        staged_info["staged"],
    )

    samples_with_eos = sum(
        1
        for row in selected
        if bool_true(row.get("eos_in_labels"))
    )
    if (
        len(selected) != TARGET_SAFE_TRAIN
        or samples_with_eos != TARGET_SAFE_TRAIN
    ):
        raise RuntimeError(
            "Không đạt đúng 30 train samples an toàn có "
            "<|SPEECH_GENERATION_END|>; dừng trước training."
        )

    train_path = dataset_dir / "train_encoded.csv"
    write_pipe_rows(train_path, selected)

    valid_path, valid_accepted, valid_rejected = (
        encode_external_validation(
            external_validation,
            run,
            tokenizer,
        )
    )

    report: dict[str, Any] = {
        "status": "prepared",
        "base_model": BASE_MODEL,
        "codec_model": CODEC_MODEL,
        "run": str(run),
        "max_len": MAX_LEN,
        "adapter_only": True,
        "source": source_meta,
        "source_train_candidates": len(train_candidates),
        "safe_after_filter_encode_audit": len(accepted_all),
        "rejected_after_train_audit": len(rejected_all),
        "final_train_count": len(selected),
        "train_speakers": len(
            {row["speakerID"] for row in selected}
        ),
        "samples_with_eos": samples_with_eos,
        "max_token_count": max(
            (
                int(row.get("total_token_count", 0) or 0)
                for row in selected
            ),
            default=0,
        ),
        "valid_source_candidates": len(external_validation),
        "valid_count": len(valid_accepted),
        "valid_rejected": len(valid_rejected),
        "valid_speakers": len(
            {
                row.get("speakerID")
                for row in valid_accepted
                if row.get("speakerID")
            }
        ),
        "dataset": str(dataset_dir),
        "train_encoded": str(train_path),
        "valid_encoded": (
            str(valid_path)
            if valid_path is not None
            else None
        ),
        "adapter": str(adapter_dir),
        "training_config": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "eos_loss_weight": args.eos_loss_weight,
            "fast_gpu": args.fast_gpu,
            "requested_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
        },
    }
    write_json(run / "training_report.json", report)

    print(
        f"✅ Dataset khóa lại: exactly {len(selected)} train | "
        f"EOS={samples_with_eos}/{TARGET_SAFE_TRAIN} | "
        f"speakers={report['train_speakers']} | "
        f"valid={len(valid_accepted)}",
        flush=True,
    )

    metrics = train_adapter(
        train_path=train_path,
        valid_path=valid_path,
        adapter_dir=adapter_dir,
        args=args,
    )

    report["status"] = "completed"
    report["training"] = metrics
    report["eos_training_report"] = {
        "eos_loss_weight": args.eos_loss_weight,
        "samples_with_eos": samples_with_eos,
        "adapter_only": True,
        "note": (
            "EOS token weighting increases the training signal for speech END; "
            "it does not mathematically guarantee that every sampled inference "
            "will emit EOS. test_lora_model.py rejects missing-EOS generations "
            "so they cannot become saved runaway audio."
        ),
    }

    write_json(run / "training_report.json", report)

    print("✅ Đã hoàn tất adapter-only LoRA.", flush=True)
    print(
        json.dumps(report, ensure_ascii=False, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
