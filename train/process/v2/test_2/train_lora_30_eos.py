"""Train a small Nghệ An LoRA without merging it into VieNeu.

This experiment intentionally lives outside the production v2 pipeline.  It
stages all eligible train candidates first, runs the official filter/codec
encoder, audits the encoded rows, and only then selects exactly 30 safe rows.
The adapter is saved separately and is loaded at inference time with PEFT.
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
from collections import defaultdict
from pathlib import Path
from typing import Any

CPU_THREADS = max(1, os.cpu_count() or 1)
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, str(CPU_THREADS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VIENEU_ENCODE_WORKERS", str(CPU_THREADS))

ROOT = Path(__file__).resolve().parents[4]
V2_DIR = ROOT / "train" / "process" / "v2"
sys.path.insert(0, str(V2_DIR))
sys.path.insert(0, str(ROOT / "source_code" / "audio_model" / "src"))
sys.path.insert(0, str(ROOT / "source_code" / "audio_model"))

from train_nghean_v2_advanced import (  # noqa: E402
    BASE_MODEL,
    MAX_LEN,
    load_source_rows,
    official_filter_and_encode,
    prepare_run_dir,
    stage_dataset,
    token_audit_and_filter,
)

DEFAULT_RUN = ROOT / "train" / "output" / "nghean_v2_lora30_eos"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train exactly 30 safe Nghe An rows as a runtime LoRA.")
    p.add_argument("--run-name", default="nghean_v2_lora30_eos")
    p.add_argument("--epochs", type=float, default=80.0)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--eos-loss-weight", type=float, default=5.0)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--workers", type=int, default=-1)
    p.add_argument("--seed", type=int, default=37)
    p.add_argument("--fast-gpu", action="store_true")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def write_pipe_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['filename']}|{row['text']}|{json.dumps(row['codes'], separators=(',', ':'))}\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def select_exactly_30(accepted: list[dict[str, Any]], staged: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer clean 5-8s rows and speakers with several utterances."""
    duration = {r["filename"]: float(r.get("duration_sec") or 0) for r in staged}
    speaker_counts: dict[str, int] = defaultdict(int)
    for row in accepted:
        speaker_counts[row["speakerID"]] += 1

    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        by_speaker[row["speakerID"]].append(row)
    for speaker in by_speaker:
        by_speaker[speaker].sort(
            key=lambda r: (
                0 if 5.0 <= duration.get(r["filename"], 0) <= 8.0 else 1,
                abs(duration.get(r["filename"], 6.0) - 6.0),
                r["filename"],
            )
        )

    speakers = sorted(by_speaker, key=lambda s: (-speaker_counts[s], s))
    selected: list[dict[str, Any]] = []
    # Give the strongest speakers a few rows each before filling globally.
    for round_index in range(4):
        for speaker in speakers:
            group = by_speaker[speaker]
            if round_index < len(group) and len(selected) < 30:
                selected.append(group[round_index])
    if len(selected) < 30:
        remaining = [r for r in accepted if r not in selected]
        remaining.sort(key=lambda r: (-speaker_counts[r["speakerID"]], r["filename"]))
        selected.extend(remaining[: 30 - len(selected)])
    if len(selected) != 30:
        raise RuntimeError(
            f"Chỉ có {len(selected)} safe encoded train samples; cần đúng 30. "
            "Không train tiếp với số lượng thiếu."
        )
    return selected


def encode_external_valid(rows: list[dict[str, Any]], run: Path, tokenizer: Any) -> tuple[Path | None, list[dict[str, Any]]]:
    if not rows:
        return None, []
    valid_dir = run / "dataset" / "external_valid"
    valid_dir.mkdir(parents=True, exist_ok=True)
    staged = stage_dataset(rows, valid_dir)
    encoded = official_filter_and_encode(valid_dir, max_samples=100000)
    audit_dir = run / "valid_eos_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    safe, accepted, _ = token_audit_and_filter(encoded, tokenizer, audit_dir)
    target = run / "dataset" / "valid_encoded.csv"
    shutil.copy2(safe, target)
    return target, accepted


class EOSWeightedTrainerMixin:
    """Token-level causal loss with extra weight only on speech END targets."""

    eos_loss_weight: float
    eos_token_id: int
    eos_seen: int
    normal_loss_sum: float
    eos_loss_sum: float
    normal_count: int
    eos_count: int

    def _init_eos_stats(self, weight: float, eos_id: int) -> None:
        self.eos_loss_weight = max(1.0, float(weight))
        self.eos_token_id = int(eos_id)
        self.eos_seen = 0
        self.normal_loss_sum = 0.0
        self.eos_loss_sum = 0.0
        self.normal_count = 0
        self.eos_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        import torch
        import torch.nn.functional as F

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )
        flat_labels = shift_labels.view(-1)
        valid = flat_labels.ne(-100)
        eos = flat_labels.eq(self.eos_token_id) & valid
        weights = torch.ones_like(losses)
        weights[eos] = self.eos_loss_weight
        valid_losses = losses[valid]
        valid_weights = weights[valid]
        loss = (valid_losses * valid_weights).sum() / valid_weights.sum().clamp_min(1.0)

        with torch.no_grad():
            self.eos_seen += int(eos.sum().item())
            self.eos_count += int(eos.sum().item())
            self.normal_count += int((valid & ~eos).sum().item())
            if (valid & ~eos).any():
                self.normal_loss_sum += float(losses[valid & ~eos].detach().float().sum().item())
            if eos.any():
                self.eos_loss_sum += float(losses[eos].detach().float().sum().item())
        return (loss, outputs) if return_outputs else loss


def train_adapter(
    train_path: Path,
    valid_path: Path | None,
    adapter_dir: Path,
    args: argparse.Namespace,
    report: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator
    from finetune.configs.lora_config import lora_config
    from finetune.train import VieNeuDataset

    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(max(1, min(4, CPU_THREADS)))
    except RuntimeError:
        pass
    cuda = torch.cuda.is_available()
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = bool(args.fast_gpu)
        torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_bf16 = bool(cuda and torch.cuda.is_bf16_supported() and not args.fp32)
    dtype = torch.float32 if args.fp32 or not cuda else torch.bfloat16 if use_bf16 else torch.float16
    device = torch.device("cuda:0" if cuda else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="sdpa" if args.fast_gpu else "eager",
    )
    model = get_peft_model(model, lora_config).to(device)
    model.config.use_cache = False
    low_vram = bool(cuda and torch.cuda.get_device_properties(0).total_memory / (1024**3) < 11)
    use_checkpointing = bool(not args.fast_gpu or low_vram)
    if use_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    class SafeDataset(VieNeuDataset):
        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            item["labels"] = item["labels"].masked_fill(item["attention_mask"] == 0, -100)
            return item

    train_ds = SafeDataset(str(train_path), tokenizer, max_len=MAX_LEN)
    valid_ds = SafeDataset(str(valid_path), tokenizer, max_len=MAX_LEN) if valid_path else None
    batch = args.batch_size
    if batch <= 0:
        batch = 4 if cuda and torch.cuda.get_device_properties(0).total_memory / (1024**3) >= 11 else 2 if cuda else 1
    workers = min(CPU_THREADS, 16) if args.workers < 0 else max(0, min(CPU_THREADS, args.workers))
    steps_per_epoch = max(1, math.ceil(len(train_ds) / max(1, batch * args.grad_accum)))
    total_steps = max(1, math.ceil(steps_per_epoch * args.epochs))
    eval_steps = max(1, total_steps // 4)
    eos_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

    class EOSTrainer(EOSWeightedTrainerMixin, Trainer):
        pass

    training_args = TrainingArguments(
        output_dir=str(adapter_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        logging_steps=max(1, total_steps // 20),
        report_to="none",
        dataloader_num_workers=workers,
        dataloader_pin_memory=cuda,
        dataloader_persistent_workers=workers > 0,
        remove_unused_columns=False,
        gradient_checkpointing=use_checkpointing,
        bf16=use_bf16,
        fp16=bool(cuda and not use_bf16 and not args.fp32),
        eval_strategy="steps" if valid_ds else "no",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=bool(valid_ds),
        metric_for_best_model="eval_loss" if valid_ds else None,
        greater_is_better=False if valid_ds else None,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = EOSTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=default_data_collator,
    )
    trainer._init_eos_stats(args.eos_loss_weight, eos_id)
    print(
        f"🦜 Train exactly {len(train_ds)} rows: {total_steps} steps, "
        f"batch={batch}, grad_accum={args.grad_accum}, eos_loss_weight={args.eos_loss_weight}",
        flush=True,
    )
    result = trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    normal_loss = trainer.normal_loss_sum / max(1, trainer.normal_count)
    eos_loss = trainer.eos_loss_sum / max(1, trainer.eos_count)
    return {
        "train_samples": len(train_ds),
        "valid_samples": len(valid_ds) if valid_ds else 0,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": batch,
        "gradient_accumulation_steps": args.grad_accum,
        "estimated_optimizer_steps": total_steps,
        "workers": workers,
        "cpu_threads": CPU_THREADS,
        "bf16": use_bf16,
        "fp32": args.fp32,
        "fast_gpu": args.fast_gpu,
        "gradient_checkpointing": use_checkpointing,
        "eos_loss_weight": args.eos_loss_weight,
        "number_of_eos_targets": trainer.eos_seen,
        "normal_token_loss": normal_loss,
        "eos_token_loss": eos_loss,
        "train_loss": float(result.training_loss) if result.training_loss is not None else None,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    run = ROOT / "train" / "output" / args.run_name
    prepare_run_dir(run, args.overwrite)
    dataset_dir = run / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print("🦜 Tìm toàn bộ candidate Nghệ An, không cắt trước 30...", flush=True)
    source_rows, source_meta = load_source_rows("all")
    train_rows = [r for r in source_rows if r["split"] == "train"]
    external_rows = [r for r in source_rows if r["split"] == "valid"] or [r for r in source_rows if r["split"] == "test"]
    if len(train_rows) < 30:
        raise RuntimeError(f"Nguồn chỉ có {len(train_rows)} train candidate, không đủ 30.")
    speaker_counts = defaultdict(int)
    for row in train_rows:
        speaker_counts[row["speakerID"]] += 1
    train_rows.sort(key=lambda r: (-speaker_counts[r["speakerID"]], 0 if 5 <= r["duration_sec"] <= 8 else 1, abs(r["duration_sec"] - 6), r["filename"]))
    staged_info = stage_dataset(train_rows, dataset_dir)
    encoded = official_filter_and_encode(dataset_dir, max_samples=100000)
    import torch
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    safe_all, accepted_all, rejected = token_audit_and_filter(encoded, tokenizer, run)
    selected = select_exactly_30(accepted_all, staged_info["staged"])
    train_path = dataset_dir / "train_encoded.csv"
    write_pipe_rows(train_path, selected)
    valid_path, valid_accepted = encode_external_valid(external_rows, run, tokenizer)
    report: dict[str, Any] = {
        "status": "prepared",
        "base_model": BASE_MODEL,
        "codec": "neuphonic/neucodec",
        "run": str(run),
        "max_len": MAX_LEN,
        "source_candidates": len(train_rows),
        "source": source_meta,
        "accepted_after_filter_encode_audit": len(accepted_all),
        "rejected_after_audit": len(rejected),
        "final_train_count": len(selected),
        "valid_count": len(valid_accepted),
        "train_speakers": len({r["speakerID"] for r in selected}),
        "valid_speakers": len({r.get("speakerID") for r in valid_accepted if r.get("speakerID")}),
        "max_token_count": max((int(r.get("total_token_count", 0) or 0) for r in selected), default=0),
        "samples_with_eos": sum(bool(r.get("eos_in_labels")) for r in selected),
        "rejected_samples": len(rejected),
        "dataset": str(dataset_dir),
        "train_encoded": str(train_path),
        "valid_encoded": str(valid_path) if valid_path else None,
        "adapter": str(run / "adapter"),
        "merged": False,
    }
    if len(selected) != 30 or report["samples_with_eos"] != 30:
        raise RuntimeError("Không đạt đúng 30 train sample có SPEECH_GENERATION_END; dừng trước train.")
    write_json(run / "training_report.json", report)
    metrics = train_adapter(train_path, valid_path, run / "adapter", args, report)
    report["status"] = "completed"
    report["training"] = metrics
    report["eos_training_report"] = {
        "eos_loss_weight": args.eos_loss_weight,
        "samples_with_eos": report["samples_with_eos"],
        "adapter_only": True,
    }
    write_json(run / "training_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
