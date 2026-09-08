"""Train exactly 30 safe Nghá»‡ An utterances as a runtime PEFT LoRA.

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
        description="Train exactly 30 safe Nghá»‡ An rows as an adapter-only VieNeu v2 LoRA."
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
    p.add_argument(
        "--fast-gpu",
        action="store_true",
        help=(
            "Fast path: BF16 + SDPA, dynamic right-padding trim, "
            "and no gradient checkpointing unless explicitly forced."
        ),
    )
    p.add_argument(
        "--force-checkpointing",
        action="store_true",
        help=(
            "Force gradient checkpointing if CUDA OOM occurs. "
            "Slower, but reduces activation memory."
        ),
    )
    p.add_argument(
        "--no-dynamic-trim",
        action="store_true",
        help=(
            "Disable batch-time trimming of right padding. "
            "Normally leave this OFF because trimming preserves valid tokens "
            "while greatly reducing compute."
        ),
    )
    p.add_argument(
        "--trim-multiple",
        type=int,
        default=8,
        help="Round dynamic sequence length up to this multiple (default 8).",
    )
    p.add_argument("--fp32", action="store_true")
    p.add_argument(
        "--validation-limit",
        type=int,
        default=12,
        help="Maximum external valid/test utterances to encode; 0 = all.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--train-only",
        action="store_true",
        help="Reuse this run's prepared dataset and start directly at LoRA training.",
    )
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


def repair_terminal_punctuation(text: str) -> tuple[str, bool]:
    """Repair staging metadata only; never modify source audio or source CSV."""
    cleaned = (text or "").strip()
    if cleaned and cleaned[-1] not in ".,?!":
        return cleaned + ".", True
    return cleaned, False


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
            f"Chá»‰ cĂ³ {len(accepted)} safe encoded train samples sau filter/encode/audit; "
            f"cáº§n Ä‘Ăºng {TARGET_SAFE_TRAIN}. KhĂ´ng train."
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


def official_filter_and_encode_local(
    dataset_dir: Path,
    max_samples: int,
) -> Path:
    """Official filter/encode without re-setting torch interop threads."""
    import torch

    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(max(1, min(4, CPU_THREADS)))
    except RuntimeError:
        # A previous encode may already have started CPU parallel work.
        pass

    from finetune.data_scripts.encode_data import encode_dataset
    from finetune.data_scripts.filter_data import filter_and_process_dataset

    filter_and_process_dataset(dataset_dir=str(dataset_dir))
    cleaned = dataset_dir / "metadata_cleaned.csv"
    if not cleaned.is_file():
        raise RuntimeError(f"Official filter did not create {cleaned}")
    encode_dataset(dataset_dir=str(dataset_dir), max_samples=max_samples)
    encoded = dataset_dir / "metadata_encoded.csv"
    if not encoded.is_file():
        raise RuntimeError(f"Official encoder did not create {encoded}")
    return encoded


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
    encoded = official_filter_and_encode_local(
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
    """Fast exact EOS-weighted causal LM loss.

    The previous test_2 implementation recomputed token-level cross entropy
    over every [batch, seq, vocab] logit in Python windows. That duplicated the
    model's own causal-LM loss work and forced gradient checkpointing on the
    9.8GB card.

    This version keeps the SAME weighted objective without recomputing CE for
    every normal token:

        weighted_loss
          = (sum(normal CE) + w * sum(EOS CE))
            / (normal_count + w * eos_count)

    The model's native loss already gives:
        base_loss = (sum(normal CE) + sum(EOS CE)) / valid_count

    Therefore we only compute an extra CE for the very small number of EOS
    positions and reconstruct the exact weighted mean. Usually there is one EOS
    target per sample, so the extra [num_eos, vocab] CE is tiny.
    """

    eos_loss_weight: float
    eos_token_id: int

    def _init_eos_stats(self, weight: float, eos_token_id: int) -> None:
        if weight <= 0:
            raise ValueError("--eos-loss-weight must be > 0")

        self.eos_loss_weight = float(weight)
        self.eos_token_id = int(eos_token_id)

        # Keep diagnostics as detached GPU scalars during training. This avoids
        # several .item() CUDA synchronizations on every optimizer micro-step.
        self._diag_eos_loss_sum = None
        self._diag_normal_loss_sum = None
        self._diag_eos_count = None
        self._diag_normal_count = None

    @staticmethod
    def _accumulate_scalar(current, value):
        value = value.detach()
        if current is None:
            return value.clone()
        current.add_(value)
        return current

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        import torch
        import torch.nn.functional as F

        labels = inputs.get("labels")
        if labels is None:
            raise RuntimeError("EOSWeightedTrainer requires labels.")

        # Let the backbone use its normal optimized causal-LM loss path.
        # This is the same fast path that made test_1 ~0.5 s/step.
        outputs = model(**inputs)
        base_loss = outputs.loss
        if base_loss is None:
            raise RuntimeError("Backbone did not return a causal LM loss.")

        shift_labels = labels[..., 1:].contiguous()
        valid_mask = shift_labels.ne(-100)
        eos_mask = shift_labels.eq(self.eos_token_id) & valid_mask

        # Counts remain tensors so there is no host/device synchronization in
        # the hot path.
        valid_count = valid_mask.sum().to(dtype=base_loss.dtype)
        eos_count = eos_mask.sum().to(dtype=base_loss.dtype)
        normal_count = valid_count - eos_count

        # Every train/valid row is EOS-audited before entering this Trainer,
        # therefore every non-empty batch contains at least one EOS target.
        # Avoid eos_mask.any().item()/bool here because that would synchronize
        # CUDA on every micro-step.
        shift_logits = outputs.logits[..., :-1, :]
        eos_logits = shift_logits[eos_mask]
        eos_targets = shift_labels[eos_mask]

        if eos_logits.shape[0] == 0:
            raise RuntimeError(
                "A Trainer batch contains zero SPEECH_GENERATION_END targets; "
                "the EOS-audited dataset invariant was broken."
            )

        # Tiny FP32 CE improves numerical stability at negligible cost.
        eos_loss_sum = F.cross_entropy(
            eos_logits.float(),
            eos_targets,
            reduction="sum",
        ).to(dtype=base_loss.dtype)

        # Recover the normal-token CE sum from the native mean loss, then
        # rebuild the exact weighted mean. For w=1 this reduces to the
        # native base loss (up to normal floating-point rounding).
        base_loss_sum = base_loss * valid_count.clamp_min(1.0)
        extra_weight = self.eos_loss_weight - 1.0
        denominator = (
            valid_count + extra_weight * eos_count
        ).clamp_min(1.0)
        loss = (
            base_loss_sum + extra_weight * eos_loss_sum
        ) / denominator

        normal_loss_sum = (
            base_loss_sum.detach() - eos_loss_sum.detach()
        )

        # Diagnostics only during training. No .item() here.
        if model.training:
            self._diag_eos_loss_sum = self._accumulate_scalar(
                self._diag_eos_loss_sum,
                eos_loss_sum,
            )
            self._diag_normal_loss_sum = self._accumulate_scalar(
                self._diag_normal_loss_sum,
                normal_loss_sum,
            )
            self._diag_eos_count = self._accumulate_scalar(
                self._diag_eos_count,
                eos_count,
            )
            self._diag_normal_count = self._accumulate_scalar(
                self._diag_normal_count,
                normal_count,
            )

        return (loss, outputs) if return_outputs else loss

    def eos_diagnostics(self) -> dict[str, float | int]:
        def scalar(value, default=0.0):
            if value is None:
                return default
            return float(value.detach().float().cpu().item())

        eos_sum = scalar(self._diag_eos_loss_sum)
        normal_sum = scalar(self._diag_normal_loss_sum)
        eos_count = scalar(self._diag_eos_count)
        normal_count = scalar(self._diag_normal_count)

        return {
            "number_of_train_eos_targets_seen": int(round(eos_count)),
            "normal_token_loss": normal_sum / max(1.0, normal_count),
            "eos_token_loss": eos_sum / max(1.0, eos_count),
        }

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
        torch.set_float32_matmul_precision(
            "high" if args.fast_gpu else "highest"
        )

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

    # Fast path: test_1 already proved batch=2 on the ~9.8GB card can train
    # without checkpointing. The old test_2 only needed forced checkpointing
    # because it recomputed full-vocab CE over every token. The new EOS loss
    # only gathers EOS positions, so restore the fast behavior.
    use_checkpointing = bool(args.force_checkpointing or not args.fast_gpu)
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

    # VieNeuDataset pads every item to MAX_LEN=2048. Most of the current
    # Nghá»‡ An utterances are far shorter. Right padding is already masked from
    # labels, so trimming ONLY the padded suffix changes no valid token, EOS
    # target, attention relation, or training objective. It simply prevents the
    # GPU from doing transformer work on hundreds/thousands of useless pads.
    trim_multiple = max(1, int(args.trim_multiple))

    def fast_dynamic_collator(features):
        batch = default_data_collator(features)
        if args.no_dynamic_trim:
            return batch

        attention = batch.get("attention_mask")
        if attention is None or attention.ndim != 2:
            return batch

        # Collation happens on CPU, so this .item() is not a CUDA sync.
        max_valid = int(attention.sum(dim=1).max().item())
        trimmed_len = max(
            trim_multiple,
            ((max_valid + trim_multiple - 1) // trim_multiple)
            * trim_multiple,
        )
        trimmed_len = min(trimmed_len, batch["input_ids"].shape[1])

        if trimmed_len < batch["input_ids"].shape[1]:
            for key in ("input_ids", "attention_mask", "labels"):
                value = batch.get(key)
                if value is not None and getattr(value, "ndim", 0) >= 2:
                    batch[key] = value[:, :trimmed_len].contiguous()

        return batch

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
        # The 9.8-GB card is now using dynamic sequence trimming and the
        # native+EOS-only loss, so batch 3 gives materially better GPU
        # occupancy than batch 2 without increasing sequence length.
        batch_size = 4 if vram_gb >= 11 else 3 if vram_gb >= 8 else 1
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
        data_collator=fast_dynamic_collator,
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
        f"đŸ¦œ Train exactly {len(train_ds)} safe rows | "
        f"valid={len(valid_ds) if valid_ds is not None else 0} | "
        f"stepsâ‰ˆ{total_steps} | batch={batch_size} | "
        f"grad_accum={args.grad_accum} | "
        f"eos_loss_weight={args.eos_loss_weight} | "
        f"dynamic_trim={not args.no_dynamic_trim} | "
        f"checkpointing={use_checkpointing}",
        flush=True,
    )

    if vram_gb is not None:
        print(
            f"đŸ¦œ Auto batch theo VRAM {vram_gb:.1f} GB: "
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

    eos_diag = trainer.eos_diagnostics()

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
            "dynamic_right_padding_trim": not args.no_dynamic_trim,
            "trim_multiple": trim_multiple,
            "eos_loss_implementation": "native_base_loss_plus_eos_only_exact_reweight",
            "eos_loss_weight": args.eos_loss_weight,
            **eos_diag,
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
    if args.trim_multiple <= 0:
        raise ValueError("--trim-multiple must be > 0")

    random.seed(args.seed)

    run = ROOT / "train" / "output" / args.run_name
    dataset_dir = run / "dataset"
    adapter_dir = run / "adapter"

    if args.train_only:
        train_path = dataset_dir / "train_encoded.csv"
        valid_path = dataset_dir / "valid_encoded.csv"
        if not train_path.is_file():
            raise FileNotFoundError(f"Missing prepared train dataset: {train_path}")
        train_count = sum(1 for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if train_count != TARGET_SAFE_TRAIN:
            raise RuntimeError(
                f"Prepared dataset has {train_count} rows; expected exactly {TARGET_SAFE_TRAIN}."
            )
        report_path = run / "training_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {
            "status": "prepared",
            "run": str(run),
            "adapter_only": True,
            "final_train_count": TARGET_SAFE_TRAIN,
        }
        print(
            f"đŸ¦œ Train-only: reuse {train_count} prepared rows; skip filter + NeuCodec encode.",
            flush=True,
        )
        metrics = train_adapter(
            train_path=train_path,
            valid_path=valid_path if valid_path.is_file() else None,
            adapter_dir=adapter_dir,
            args=args,
        )
        report["status"] = "completed"
        report["training"] = metrics
        write_json(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return

    prepare_run_dir(run, args.overwrite)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(
        "đŸ¦œ TĂ¬m toĂ n bá»™ candidate Nghá»‡ An; "
        "khĂ´ng cáº¯t xuá»‘ng 30 trÆ°á»›c filter/encode...",
        flush=True,
    )

    source_rows, source_meta = load_source_rows("all")

    train_candidates = [
        row
        for row in source_rows
        if row["split"] == "train"
    ]
    repaired_transcripts = 0
    for row in train_candidates:
        repaired, changed = repair_terminal_punctuation(row["transcript"])
        row["transcript"] = repaired
        repaired_transcripts += int(changed)
    external_validation = choose_external_validation_rows(
        source_rows,
        args.validation_limit,
    )

    if len(train_candidates) < TARGET_SAFE_TRAIN:
        raise RuntimeError(
            f"Nguá»“n chá»‰ cĂ³ {len(train_candidates)} eligible train candidates; "
            f"cáº§n Ă­t nháº¥t {TARGET_SAFE_TRAIN}."
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
        f"đŸ¦œ Staged {len(train_candidates)} train candidates; "
        "cháº¡y official filter + NeuCodec encode...",
        flush=True,
    )

    encoded = official_filter_and_encode_local(
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
        f"đŸ¦œ Safe sau filter/encode/token audit: "
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
            "KhĂ´ng Ä‘áº¡t Ä‘Ăºng 30 train samples an toĂ n cĂ³ "
            "<|SPEECH_GENERATION_END|>; dá»«ng trÆ°á»›c training."
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
        "staging_transcripts_terminal_punctuation_repaired": repaired_transcripts,
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
            "force_checkpointing": args.force_checkpointing,
            "dynamic_right_padding_trim": not args.no_dynamic_trim,
            "trim_multiple": max(1, int(args.trim_multiple)),
            "requested_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
        },
    }
    write_json(run / "training_report.json", report)

    print(
        f"âœ… Dataset khĂ³a láº¡i: exactly {len(selected)} train | "
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

    print("âœ… ÄĂ£ hoĂ n táº¥t adapter-only LoRA.", flush=True)
    print(
        json.dumps(report, ensure_ascii=False, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()

