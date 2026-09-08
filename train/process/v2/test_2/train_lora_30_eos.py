"""Aggressive Nghá»‡ An accent + EOS-focused PEFT LoRA training.

Goals
-----
- Keep the verified v2 preprocessing/filter/NeuCodec/token-audit pipeline.
- Keep EXACTLY 30 already-safe training rows so this remains directly
  comparable with test_2.
- Push regional pronunciation much harder than the previous LoRA:
  * larger LoRA rank/alpha,
  * zero LoRA dropout,
  * LoRA also on lm_head,
  * stronger learning rate + cosine schedule,
  * loss only on generated speech codes + END (not the generation-start token).
- Reduce missing EOS without encouraging premature stopping:
  * strong END weight,
  * END weight ramps up late in training,
  * extra supervision on the final speech-code window,
  * anti-early-EOS margin penalty before the true END.
- Preserve the fast single-GPU / multi-CPU path:
  BF16 + SDPA + dynamic right-padding trim + no checkpointing on --fast-gpu.
- Save adapter only. Never merge the base model.

This file does not modify source_code/audio_model or the inference/test file.
"""

from __future__ import annotations

import argparse
import copy
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
DEFAULT_RUN_NAME = "nghean_v2_lora30_accentmax_eos"
TARGET_SAFE_TRAIN = 30

# Aggressive accent preset. The official config is r=16, alpha=32, dropout=0.05.
DEFAULT_LEARNING_RATE = 1.2e-5
DEFAULT_LORA_R = 32
DEFAULT_LORA_ALPHA = 96
DEFAULT_LORA_DROPOUT = 0.0

# EOS preset: start moderately, then become much stronger late in training.
DEFAULT_EOS_START_WEIGHT = 6.0
DEFAULT_EOS_LOSS_WEIGHT = 18.0
DEFAULT_EOS_RAMP_START = 0.55
DEFAULT_EOS_TAIL_TOKENS = 64
DEFAULT_EOS_TAIL_WEIGHT = 1.35
DEFAULT_EARLY_EOS_PENALTY = 0.08
DEFAULT_EARLY_EOS_MARGIN = 0.5


# ---------------------------------------------------------------------------
# CLI / simple IO
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Aggressive Nghá»‡ An accent + EOS-focused adapter-only VieNeu v2 LoRA."
        )
    )
    p.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    p.add_argument("--epochs", type=float, default=60.0)
    p.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)

    # Reuse an already-prepared dataset from another run while writing a new
    # adapter to --run-name. This lets test_2 keep its old dataset untouched.
    p.add_argument(
        "--dataset-run-name",
        default=None,
        help=(
            "With --train-only, read dataset/train_encoded.csv and "
            "valid_encoded.csv from this run name. If omitted, use --run-name."
        ),
    )

    # Stronger LoRA capacity for pronunciation/acoustic-token adaptation.
    p.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    p.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    p.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    p.add_argument(
        "--no-lm-head-lora",
        action="store_true",
        help=(
            "Disable LoRA on lm_head. Default keeps it ON because adapting the "
            "speech-token output projection materially increases accent/EOS capacity."
        ),
    )

    # EOS objective.
    p.add_argument(
        "--eos-loss-weight",
        type=float,
        default=DEFAULT_EOS_LOSS_WEIGHT,
        help="Final END-token weight used late in training and for evaluation.",
    )
    p.add_argument(
        "--eos-start-weight",
        type=float,
        default=DEFAULT_EOS_START_WEIGHT,
        help="END-token weight before the late-training ramp begins.",
    )
    p.add_argument(
        "--eos-ramp-start",
        type=float,
        default=DEFAULT_EOS_RAMP_START,
        help="Training-progress fraction at which END weight starts ramping to final.",
    )
    p.add_argument(
        "--eos-tail-tokens",
        type=int,
        default=DEFAULT_EOS_TAIL_TOKENS,
        help="Number of speech-code targets immediately before END to up-weight.",
    )
    p.add_argument(
        "--eos-tail-weight",
        type=float,
        default=DEFAULT_EOS_TAIL_WEIGHT,
        help="Weight for the final speech-code window before END.",
    )
    p.add_argument(
        "--early-eos-penalty",
        type=float,
        default=DEFAULT_EARLY_EOS_PENALTY,
        help="Margin penalty that keeps END below the true next token before the boundary.",
    )
    p.add_argument(
        "--early-eos-margin",
        type=float,
        default=DEFAULT_EARLY_EOS_MARGIN,
    )

    p.add_argument(
        "--scheduler",
        choices=("cosine", "linear", "constant_with_warmup"),
        default="cosine",
        help="cosine is the recommended aggressive-but-stable accent schedule.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 = auto by VRAM: >=11GB -> 4, >=8GB -> 3, otherwise 1.",
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
        help="Force gradient checkpointing if CUDA OOM occurs.",
    )
    p.add_argument(
        "--no-dynamic-trim",
        action="store_true",
        help="Disable trimming of masked right padding at batch time.",
    )
    p.add_argument(
        "--trim-multiple",
        type=int,
        default=8,
        help="Round dynamic sequence length up to this multiple.",
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
        help="Reuse a prepared encoded dataset and start directly at LoRA training.",
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
    """Choose exactly 30 safe rows with stronger accent concentration.

    Preference:
    1) speakers with multiple usable utterances first;
    2) take several utterances from the same repeated speaker before moving on;
    3) prefer 6-12 second utterances (more phonetic/context coverage);
    4) fill with singletons only when necessary.

    If the source itself contains almost one utterance per speaker, this cannot
    invent repeated-speaker data; it simply avoids maximizing speaker diversity.
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
                if 6.0 <= duration_by_filename.get(row["filename"], 0.0) <= 12.0
                else 1,
                abs(duration_by_filename.get(row["filename"], 8.0) - 8.0),
                int(row.get("total_token_count", 0) or 0),
                row["filename"],
            )
        )

    speakers = sorted(
        by_speaker,
        key=lambda spk: (-len(by_speaker[spk]), spk),
    )

    selected: list[dict[str, Any]] = []

    # Accent-concentrated pass: repeated speakers contribute up to 8 rows before
    # singleton speakers are needed.
    for speaker in speakers:
        if len(by_speaker[speaker]) <= 1:
            continue
        for row in by_speaker[speaker][:8]:
            if len(selected) >= TARGET_SAFE_TRAIN:
                break
            selected.append(row)
        if len(selected) >= TARGET_SAFE_TRAIN:
            break

    # Fill from all remaining safe rows, still preferring repeated speakers and
    # useful duration.
    if len(selected) < TARGET_SAFE_TRAIN:
        already = {id(row) for row in selected}
        speaker_counts = {spk: len(rows) for spk, rows in by_speaker.items()}
        remaining = [row for row in accepted if id(row) not in already]
        remaining.sort(
            key=lambda row: (
                -speaker_counts[row["speakerID"]],
                0
                if 6.0 <= duration_by_filename.get(row["filename"], 0.0) <= 12.0
                else 1,
                abs(duration_by_filename.get(row["filename"], 8.0) - 8.0),
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
    """Fast endpoint-aware loss for stronger END behavior.

    The official VieNeu dataset already supervises the generated speech region.
    This trainer keeps the fast native causal-LM loss, then adds only small
    targeted losses:
      - stronger CE on the true SPEECH_GENERATION_END position;
      - mild CE boost on the final N speech-code targets before END;
      - margin penalty that prevents END from outranking the correct next code
        before the actual boundary.

    No full [batch, seq, vocab] per-token CE tensor is materialized.
    """

    eos_token_id: int

    def _init_eos_stats(
        self,
        final_weight: float,
        start_weight: float,
        ramp_start: float,
        tail_tokens: int,
        tail_weight: float,
        early_eos_penalty: float,
        early_eos_margin: float,
        eos_token_id: int,
    ) -> None:
        if final_weight <= 0 or start_weight <= 0:
            raise ValueError("EOS weights must be > 0")
        if not 0.0 <= ramp_start < 1.0:
            raise ValueError("--eos-ramp-start must be in [0, 1)")
        if tail_tokens < 0:
            raise ValueError("--eos-tail-tokens must be >= 0")
        if tail_weight < 1.0:
            raise ValueError("--eos-tail-weight must be >= 1")
        if early_eos_penalty < 0:
            raise ValueError("--early-eos-penalty must be >= 0")

        self.eos_final_weight = float(final_weight)
        self.eos_start_weight = float(start_weight)
        self.eos_ramp_start = float(ramp_start)
        self.eos_tail_tokens = int(tail_tokens)
        self.eos_tail_weight = float(tail_weight)
        self.early_eos_penalty = float(early_eos_penalty)
        self.early_eos_margin = float(early_eos_margin)
        self.eos_token_id = int(eos_token_id)

        self._diag_eos_loss_sum = None
        self._diag_normal_loss_sum = None
        self._diag_tail_loss_sum = None
        self._diag_early_eos_penalty_sum = None
        self._diag_eos_count = None
        self._diag_normal_count = None
        self._diag_tail_count = None
        self._diag_batches = None

    @staticmethod
    def _accumulate_scalar(current, value):
        value = value.detach()
        if current is None:
            return value.clone()
        current.add_(value)
        return current

    def _current_eos_weight(self, training: bool) -> float:
        # Evaluation always uses the final weight, so eval losses at different
        # checkpoints remain comparable.
        if not training:
            return self.eos_final_weight

        max_steps = max(1, int(getattr(self.state, "max_steps", 1) or 1))
        step = max(0, int(getattr(self.state, "global_step", 0) or 0))
        progress = min(1.0, step / max_steps)

        if progress <= self.eos_ramp_start:
            return self.eos_start_weight

        t = (
            (progress - self.eos_ramp_start)
            / max(1e-8, 1.0 - self.eos_ramp_start)
        )
        return (
            self.eos_start_weight
            + t * (self.eos_final_weight - self.eos_start_weight)
        )

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

        outputs = model(**inputs)
        base_loss = outputs.loss
        if base_loss is None:
            raise RuntimeError("Backbone did not return a causal LM loss.")

        shift_labels = labels[..., 1:].contiguous()
        shift_logits = outputs.logits[..., :-1, :]

        valid_mask = shift_labels.ne(-100)
        eos_mask = shift_labels.eq(self.eos_token_id) & valid_mask

        valid_count = valid_mask.sum().to(dtype=base_loss.dtype)
        eos_count = eos_mask.sum().to(dtype=base_loss.dtype)
        normal_count = valid_count - eos_count

        eos_logits = shift_logits[eos_mask]
        eos_targets = shift_labels[eos_mask]
        if eos_logits.shape[0] == 0:
            raise RuntimeError(
                "A batch contains zero SPEECH_GENERATION_END targets; "
                "the EOS-audited dataset invariant was broken."
            )

        eos_loss_sum = F.cross_entropy(
            eos_logits.float(),
            eos_targets,
            reduction="sum",
        ).to(dtype=base_loss.dtype)

        # Final N speech-code positions before the true EOS.
        seq_len = shift_labels.shape[1]
        positions = torch.arange(
            seq_len,
            device=shift_labels.device,
        ).unsqueeze(0)

        eos_pos_candidates = torch.where(
            eos_mask,
            positions,
            torch.full_like(positions, seq_len),
        )
        eos_pos = eos_pos_candidates.min(dim=1).values
        tail_start = (eos_pos - self.eos_tail_tokens).clamp_min(0)

        tail_mask = (
            valid_mask
            & ~eos_mask
            & (positions >= tail_start.unsqueeze(1))
            & (positions < eos_pos.unsqueeze(1))
        )

        tail_count = tail_mask.sum().to(dtype=base_loss.dtype)
        if self.eos_tail_tokens > 0 and tail_mask.shape[0] > 0:
            tail_logits = shift_logits[tail_mask]
            tail_targets = shift_labels[tail_mask]
        else:
            tail_logits = shift_logits.new_empty(
                (0, shift_logits.shape[-1])
            )
            tail_targets = shift_labels.new_empty((0,))

        if tail_logits.shape[0] > 0:
            tail_loss_sum = F.cross_entropy(
                tail_logits.float(),
                tail_targets,
                reduction="sum",
            ).to(dtype=base_loss.dtype)

            # Before the real boundary, END should not beat the correct next
            # speech code. This counterbalances the stronger END weight.
            correct_logits = tail_logits.gather(
                1,
                tail_targets.unsqueeze(1),
            ).squeeze(1)
            premature_eos_logits = tail_logits[:, self.eos_token_id]
            boundary_penalty = F.softplus(
                premature_eos_logits.float()
                - correct_logits.float()
                + self.early_eos_margin
            ).mean().to(dtype=base_loss.dtype)
        else:
            tail_loss_sum = base_loss.detach() * 0.0
            boundary_penalty = base_loss.detach() * 0.0

        base_loss_sum = base_loss * valid_count.clamp_min(1.0)

        current_eos_weight = self._current_eos_weight(model.training)
        eos_extra = current_eos_weight - 1.0
        tail_extra = self.eos_tail_weight - 1.0

        weighted_sum = (
            base_loss_sum
            + eos_extra * eos_loss_sum
            + tail_extra * tail_loss_sum
        )
        weighted_count = (
            valid_count
            + eos_extra * eos_count
            + tail_extra * tail_count
        ).clamp_min(1.0)

        loss = weighted_sum / weighted_count
        if self.early_eos_penalty > 0:
            loss = loss + self.early_eos_penalty * boundary_penalty

        normal_loss_sum = (
            base_loss_sum.detach() - eos_loss_sum.detach()
        )

        if model.training:
            one = eos_count.detach() * 0.0 + 1.0
            self._diag_eos_loss_sum = self._accumulate_scalar(
                self._diag_eos_loss_sum, eos_loss_sum
            )
            self._diag_normal_loss_sum = self._accumulate_scalar(
                self._diag_normal_loss_sum, normal_loss_sum
            )
            self._diag_tail_loss_sum = self._accumulate_scalar(
                self._diag_tail_loss_sum, tail_loss_sum
            )
            self._diag_early_eos_penalty_sum = self._accumulate_scalar(
                self._diag_early_eos_penalty_sum, boundary_penalty
            )
            self._diag_eos_count = self._accumulate_scalar(
                self._diag_eos_count, eos_count
            )
            self._diag_normal_count = self._accumulate_scalar(
                self._diag_normal_count, normal_count
            )
            self._diag_tail_count = self._accumulate_scalar(
                self._diag_tail_count, tail_count
            )
            self._diag_batches = self._accumulate_scalar(
                self._diag_batches, one
            )

        return (loss, outputs) if return_outputs else loss

    def eos_diagnostics(self) -> dict[str, float | int]:
        def scalar(value, default=0.0):
            if value is None:
                return default
            return float(value.detach().float().cpu().item())

        eos_sum = scalar(self._diag_eos_loss_sum)
        normal_sum = scalar(self._diag_normal_loss_sum)
        tail_sum = scalar(self._diag_tail_loss_sum)
        boundary_sum = scalar(self._diag_early_eos_penalty_sum)
        eos_count = scalar(self._diag_eos_count)
        normal_count = scalar(self._diag_normal_count)
        tail_count = scalar(self._diag_tail_count)
        batches = scalar(self._diag_batches)

        return {
            "number_of_train_eos_targets_seen": int(round(eos_count)),
            "normal_token_loss": normal_sum / max(1.0, normal_count),
            "eos_token_loss": eos_sum / max(1.0, eos_count),
            "tail_token_loss": tail_sum / max(1.0, tail_count),
            "average_early_eos_margin_penalty": (
                boundary_sum / max(1.0, batches)
            ),
            "eos_start_weight": self.eos_start_weight,
            "eos_final_weight": self.eos_final_weight,
            "eos_tail_tokens": self.eos_tail_tokens,
            "eos_tail_weight": self.eos_tail_weight,
            "early_eos_penalty": self.early_eos_penalty,
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

    speech_start_id = tokenizer.convert_tokens_to_ids(
        "<|SPEECH_GENERATION_START|>"
    )
    speech_end_id = tokenizer.convert_tokens_to_ids(
        "<|SPEECH_GENERATION_END|>"
    )
    if speech_start_id is None or int(speech_start_id) < 0:
        raise RuntimeError(
            "Base tokenizer does not expose <|SPEECH_GENERATION_START|>."
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

    # Strong accent LoRA:
    # official = r16 / alpha32 / dropout0.05 on transformer projections.
    # default here = r32 / alpha96 / dropout0.0 + lm_head LoRA.
    accent_lora_config = copy.deepcopy(lora_config)
    accent_lora_config.r = int(args.lora_r)
    accent_lora_config.lora_alpha = int(args.lora_alpha)
    accent_lora_config.lora_dropout = float(args.lora_dropout)

    target_modules = list(accent_lora_config.target_modules or [])
    if not args.no_lm_head_lora and "lm_head" not in target_modules:
        target_modules.append("lm_head")
    accent_lora_config.target_modules = target_modules

    model = get_peft_model(model, accent_lora_config)
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
        """Speech-code-focused labels + safe padding mask.

        Official VieNeu already masks the text prompt and starts labels at
        SPEECH_GENERATION_START. We additionally mask that START token itself,
        so every supervised non-EOS token is an acoustic speech code. We also
        mask anything after the first true speech END for strict boundary focus.
        """

        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            labels = item["labels"].clone()
            input_ids = item["input_ids"]

            labels = labels.masked_fill(
                item["attention_mask"] == 0,
                -100,
            )

            start_positions = (
                input_ids == int(speech_start_id)
            ).nonzero(as_tuple=True)[0]
            end_positions = (
                input_ids == int(speech_end_id)
            ).nonzero(as_tuple=True)[0]

            if len(start_positions) == 0 or len(end_positions) == 0:
                raise RuntimeError(
                    f"Sample {idx} lost speech START/END after dataset loading."
                )

            start_pos = int(start_positions[0])
            end_pos = int(end_positions[0])
            if end_pos <= start_pos:
                raise RuntimeError(
                    f"Sample {idx} has invalid speech boundary ordering."
                )

            labels[: start_pos + 1] = -100
            if end_pos + 1 < labels.shape[0]:
                labels[end_pos + 1 :] = -100

            item["labels"] = labels
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
    eval_steps = max(1, total_steps // 6)
    save_steps = eval_steps

    has_eval = valid_ds is not None and len(valid_ds) > 0

    training_kwargs: dict[str, Any] = dict(
        output_dir=str(adapter_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type=args.scheduler,
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
            EarlyStoppingCallback(early_stopping_patience=4)
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
        final_weight=args.eos_loss_weight,
        start_weight=args.eos_start_weight,
        ramp_start=args.eos_ramp_start,
        tail_tokens=args.eos_tail_tokens,
        tail_weight=args.eos_tail_weight,
        early_eos_penalty=args.early_eos_penalty,
        early_eos_margin=args.early_eos_margin,
        eos_token_id=int(speech_end_id),
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
        f"EOS={args.eos_start_weight}->{args.eos_loss_weight} | "
        f"LoRA=r{args.lora_r}/a{args.lora_alpha} | "
        f"lm_head_lora={not args.no_lm_head_lora} | "
        f"scheduler={args.scheduler} | "
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
            "scheduler": args.scheduler,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lm_head_lora": not args.no_lm_head_lora,
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
            "eos_loss_implementation": (
                "native_speech_loss_plus_dynamic_eos_tail_boundary_margin"
            ),
            "eos_loss_weight": args.eos_loss_weight,
            "eos_start_weight": args.eos_start_weight,
            "eos_ramp_start": args.eos_ramp_start,
            "eos_tail_tokens": args.eos_tail_tokens,
            "eos_tail_weight": args.eos_tail_weight,
            "early_eos_penalty": args.early_eos_penalty,
            "early_eos_margin": args.early_eos_margin,
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
    if args.lora_r <= 0:
        raise ValueError("--lora-r must be > 0")
    if args.lora_alpha <= 0:
        raise ValueError("--lora-alpha must be > 0")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora-dropout must be in [0, 1)")
    if args.eos_start_weight <= 0:
        raise ValueError("--eos-start-weight must be > 0")
    if not 0.0 <= args.eos_ramp_start < 1.0:
        raise ValueError("--eos-ramp-start must be in [0, 1)")
    if args.eos_tail_tokens < 0:
        raise ValueError("--eos-tail-tokens must be >= 0")
    if args.eos_tail_weight < 1.0:
        raise ValueError("--eos-tail-weight must be >= 1")
    if args.early_eos_penalty < 0:
        raise ValueError("--early-eos-penalty must be >= 0")

    random.seed(args.seed)

    run = ROOT / "train" / "output" / args.run_name
    dataset_dir = run / "dataset"
    adapter_dir = run / "adapter"

    if args.train_only:
        dataset_source_run_name = args.dataset_run_name or args.run_name
        dataset_source_run = (
            ROOT / "train" / "output" / dataset_source_run_name
        )
        dataset_source_dir = dataset_source_run / "dataset"

        train_path = dataset_source_dir / "train_encoded.csv"
        valid_path = dataset_source_dir / "valid_encoded.csv"

        if not train_path.is_file():
            raise FileNotFoundError(
                f"Missing prepared train dataset: {train_path}"
            )

        train_count = sum(
            1
            for line in train_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )
        if train_count != TARGET_SAFE_TRAIN:
            raise RuntimeError(
                f"Prepared dataset has {train_count} rows; "
                f"expected exactly {TARGET_SAFE_TRAIN}."
            )

        run.mkdir(parents=True, exist_ok=True)

        # --overwrite in train-only mode removes ONLY this output adapter,
        # never the source encoded dataset.
        if (
            args.overwrite
            and adapter_dir.exists()
            and args.resume_from_checkpoint is None
        ):
            shutil.rmtree(adapter_dir)

        source_report_path = dataset_source_run / "training_report.json"
        report_path = run / "training_report.json"

        if report_path.is_file():
            report = json.loads(
                report_path.read_text(encoding="utf-8")
            )
        elif source_report_path.is_file():
            source_report = json.loads(
                source_report_path.read_text(encoding="utf-8")
            )
            report = {
                "status": "prepared",
                "base_model": source_report.get("base_model", BASE_MODEL),
                "codec_model": source_report.get("codec_model", CODEC_MODEL),
                "run": str(run),
                "adapter_only": True,
                "final_train_count": TARGET_SAFE_TRAIN,
                "dataset_reused_from": str(dataset_source_run),
                "source": source_report.get("source"),
                "train_speakers": source_report.get("train_speakers"),
                "samples_with_eos": source_report.get("samples_with_eos"),
                "max_token_count": source_report.get("max_token_count"),
                "valid_count": source_report.get("valid_count"),
            }
        else:
            report = {
                "status": "prepared",
                "run": str(run),
                "adapter_only": True,
                "final_train_count": TARGET_SAFE_TRAIN,
                "dataset_reused_from": str(dataset_source_run),
            }

        report["training_config"] = {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "scheduler": args.scheduler,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lm_head_lora": not args.no_lm_head_lora,
            "eos_start_weight": args.eos_start_weight,
            "eos_loss_weight": args.eos_loss_weight,
            "eos_ramp_start": args.eos_ramp_start,
            "eos_tail_tokens": args.eos_tail_tokens,
            "eos_tail_weight": args.eos_tail_weight,
            "early_eos_penalty": args.early_eos_penalty,
            "fast_gpu": args.fast_gpu,
            "requested_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
        }

        print(
            f"đŸ¦œ Train-only: reuse {train_count} prepared rows from "
            f"{dataset_source_run}; skip filter + NeuCodec encode.",
            flush=True,
        )

        metrics = train_adapter(
            train_path=train_path,
            valid_path=valid_path if valid_path.is_file() else None,
            adapter_dir=adapter_dir,
            args=args,
        )

        report["status"] = "completed"
        report["adapter"] = str(adapter_dir)
        report["training"] = metrics
        write_json(report_path, report)
        print(
            json.dumps(report, ensure_ascii=False, indent=2),
            flush=True,
        )
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
            "scheduler": args.scheduler,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lm_head_lora": not args.no_lm_head_lora,
            "eos_start_weight": args.eos_start_weight,
            "eos_loss_weight": args.eos_loss_weight,
            "eos_ramp_start": args.eos_ramp_start,
            "eos_tail_tokens": args.eos_tail_tokens,
            "eos_tail_weight": args.eos_tail_weight,
            "early_eos_penalty": args.early_eos_penalty,
            "early_eos_margin": args.early_eos_margin,
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
            "This run uses a strong late END ramp, final-code tail weighting, "
            "and an anti-early-END margin. These increase endpoint supervision "
            "but still cannot mathematically guarantee EOS on every sampled "
            "inference. The existing test_lora_model.py remains unchanged."
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

