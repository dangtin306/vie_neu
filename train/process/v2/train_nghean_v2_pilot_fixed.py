"""Validated Nghệ An LoRA pilot.

This is intentionally a new pipeline.  The original pilot is kept untouched.
It audits the exact VieNeu preprocessing format before training, uses a
speaker-disjoint validation split, runs a baseline first, and records EOS/A-B
results in a separate output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

CPU_THREADS = max(1, os.cpu_count() or 1)
DATA_WORKERS = CPU_THREADS
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "source_code" / "audio_model"
sys.path.insert(0, str(SOURCE_ROOT / "src"))
sys.path.insert(0, str(SOURCE_ROOT))

PREPARED = ROOT / "train" / "output" / "nghean_test2_accent_scale"
MANIFEST = PREPARED / "nghean_eligible_audio_manifest.csv"
RUN = ROOT / "train" / "output" / "nghean_v2_pilot_fixed"
DATASET = RUN / "pilot_dataset"
ADAPTER = RUN / "adapter"
MERGED = RUN / "merged_model"
BASE = "pnnbao-ump/VieNeu-TTS-0.3B"
SEED = 37
TEMPERATURE = 0.35
TOP_K = 25
MAX_LEN = 2048
TEXTS = [
    "Hôm nay tôi đang thử nghiệm giọng nói tiếng Việt.",
    "Tôi xin chào quý vị và các bạn.",
    "Sáng nay trời mát, mọi người đi làm rất sớm.",
    "Chương trình hôm nay có nhiều thông tin đáng chú ý.",
    "Đây là bản tin được thực hiện bằng giọng nói nhân tạo.",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-speakers", type=int, default=20)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def read_manifest():
    with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not any(r.get("split") == "valid" for r in rows):
        with (ROOT / "train" / "metadata_na_candidates.csv").open(encoding="utf-8-sig", newline="") as f:
            rows.extend(r for r in csv.DictReader(f) if r.get("split") == "valid")
    clean = []
    seen = set()
    for row in rows:
        path = Path(row["local_path"])
        if not path.is_absolute():
            path = ROOT / "train" / path
        text = (row.get("transcript") or row.get("text") or "").strip()
        try:
            duration = float(row.get("duration_sec") or row.get("duration") or row.get("duration_metadata") or 0)
        except ValueError:
            duration = 0.0
        key = str(path.resolve())
        qc = (row.get("qc_status") or "pass").lower()
        if (row.get("split") not in {"train", "test", "valid"} or row.get("status", "clean").lower() in {"reject", "bad"}
                or qc not in {"pass", "clean", "acceptable"}
                or not text or not path.is_file() or key in seen):
            continue
        seen.add(key)
        row["local_path"] = str(path)
        row["transcript"] = text
        row["duration_sec"] = f"{duration:.3f}"
        clean.append(row)
    return clean


def choose_rows(seed: int, n_speakers: int):
    rows = read_manifest()
    train = [r for r in rows if r["split"] == "train"]
    if not train:
        raise RuntimeError("Need clean train rows")
    counts = Counter(r["speakerID"] for r in train)
    speakers = [s for s, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:n_speakers]]
    selected = [r for r in train if r["speakerID"] in speakers]
    # Prefer the requested 4-10 second window, but do not randomly truncate the
    # selected speakers.  Every selected row is later checked by token audit.
    selected.sort(key=lambda r: (0 if 4 <= float(r["duration_sec"]) <= 10 else 1,
                                 r["speakerID"], r["filename"]))
    # A held-out speaker may come from test/valid or a clean row of a speaker
    # intentionally excluded from train. The speaker is the isolation boundary.
    test = [r for r in rows if r["speakerID"] not in set(speakers)
            and r["transcript"].strip() and float(r["duration_sec"]) <= 10]
    if not test:
        raise RuntimeError("No held-out test speaker remains outside the train speakers")
    # Candidate order: preferred 4-8 sec, fallback 8-10 sec, then filename.
    test.sort(key=lambda r: (0 if 4 <= float(r["duration_sec"]) <= 8 else 1,
                             0 if r["split"] == "test" else 1 if r["split"] == "valid" else 2,
                             float(r["duration_sec"]), r["speakerID"], r["filename"]))
    unique = {}
    for row in test:
        unique.setdefault(row["speakerID"], row)
    test = list(unique.values())
    rng = random.Random(seed)
    speaker_order = list(speakers)
    rng.shuffle(speaker_order)
    n_valid = max(1, round(len(speaker_order) * 0.15))
    valid_speakers = set(speaker_order[:n_valid])
    train_speakers = set(speaker_order[n_valid:])
    if not train_speakers:
        raise RuntimeError("Speaker split left no training speakers")
    return selected, test, train_speakers, valid_speakers


def stage(rows):
    if DATASET.exists():
        shutil.rmtree(DATASET)
    raw = DATASET / "raw_audio"
    raw.mkdir(parents=True)
    staged = []
    for row in rows:
        name = f"{row['speakerID']}__{row['filename']}"
        shutil.copy2(row["local_path"], raw / name)
        staged.append((name, row["transcript"], row["speakerID"], row["duration_sec"]))
    with (DATASET / "metadata.csv").open("w", encoding="utf-8") as f:
        for name, text, *_ in staged:
            f.write(f"{name}|{text}\n")
    with (DATASET / "metadata_cleaned.csv").open("w", encoding="utf-8") as f:
        for name, text, *_ in staged:
            f.write(f"{name}|{text}\n")
    return staged


def encode():
    from finetune.data_scripts.encode_data import encode_dataset
    encode_dataset(dataset_dir=str(DATASET), max_samples=100000)
    encoded = DATASET / "metadata_encoded.csv"
    if not encoded.is_file():
        raise RuntimeError("NeuCodec did not create metadata_encoded.csv")
    return encoded


def audit(encoded, staged, tokenizer):
    from vieneu_utils.phonemize_text import phonemize_with_dict
    rows = []
    staged_map = {x[0]: x for x in staged}
    with encoded.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|", 2)
            if len(parts) != 3 or parts[0] not in staged_map:
                continue
            filename, text, codes_json = parts
            codes = json.loads(codes_json)
            speaker, _, _ = filename.partition("__")
            phones = phonemize_with_dict(text)
            chat = (f"<|TEXT_PROMPT_START|>{phones}<|TEXT_PROMPT_END|>"
                    f"<|SPEECH_GENERATION_START|>{''.join(f'<|speech_{i}|>' for i in codes)}"
                    f"<|SPEECH_GENERATION_END|>")
            ids = tokenizer.encode(chat)
            start = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
            end = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
            start_pos = ids.index(start) if start in ids else -1
            end_pos = ids.index(end) if end in ids else -1
            rows.append({
                "filename": filename, "speakerID": speaker,
                "duration_sec": staged_map[filename][3],
                "phoneme_length": len(tokenizer.encode(phones, add_special_tokens=False)),
                "speech_code_count": len(codes), "total_token_count": len(ids),
                "has_speech_start": start_pos >= 0, "has_speech_end": end_pos >= 0,
                "eos_in_labels": start_pos >= 0 and end_pos >= start_pos,
                "over_2048": len(ids) > MAX_LEN,
                "reason": "" if start_pos >= 0 and end_pos >= start_pos and len(ids) <= MAX_LEN else "invalid_or_context_overflow",
                "text": text, "codes": codes,
            })
    with (RUN / "dataset_token_audit.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["filename", "speakerID", "duration_sec", "phoneme_length", "speech_code_count", "total_token_count",
                  "has_speech_start", "has_speech_end", "eos_in_labels", "over_2048", "reason"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows([{k: r[k] for k in fields} for r in rows])
    accepted = [r for r in rows if r["has_speech_start"] and r["has_speech_end"] and r["eos_in_labels"] and not r["over_2048"]]
    if not rows or len(accepted) != len(rows):
        raise RuntimeError(f"Token audit rejected {len(rows) - len(accepted)} samples; no truncation is allowed")
    if not all(r["has_speech_start"] and r["has_speech_end"] and r["eos_in_labels"] and not r["over_2048"] for r in accepted):
        raise RuntimeError("Token audit assertions failed")
    return rows


def write_split_metadata(audited, train_speakers, valid_speakers):
    by_speaker = {r["speakerID"]: r for r in audited}
    train = [r for r in audited if r["speakerID"] in train_speakers]
    valid = [r for r in audited if r["speakerID"] in valid_speakers]
    if not train or not valid:
        raise RuntimeError("Empty train or validation split")
    paths = {}
    for name, subset in (("train_encoded.csv", train), ("valid_encoded.csv", valid)):
        path = DATASET / name
        with path.open("w", encoding="utf-8") as f:
            for r in subset:
                f.write(f"{r['filename']}|{r['text']}|{json.dumps(r['codes'], separators=(',', ':'))}\n")
        paths[name] = path
    return paths, train, valid


def infer(engine, text, ref_audio, ref_text, output):
    import torch
    import numpy as np
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    audio = engine.infer(text, ref_audio=str(ref_audio), ref_text=ref_text, max_chars=256,
                         temperature=TEMPERATURE, top_k=TOP_K, apply_watermark=False)
    engine.save(audio, str(output))
    return float(len(audio) / engine.sample_rate)


def baseline(test_row, output_dir, texts=TEXTS, start_index=1):
    from vieneu import Vieneu
    engine = Vieneu(mode="standard", backbone_repo=BASE, backbone_device="cuda",
                    codec_repo="neuphonic/neucodec", codec_device="cuda", gguf_filename=None)
    out = output_dir; out.mkdir(parents=True, exist_ok=True)
    durations = []
    for i, text in enumerate(texts, start_index):
        durations.append(infer(engine, text, Path(test_row["local_path"]), test_row["transcript"], out / f"base_{i:02d}.wav"))
    engine.close()
    warnings = []
    if any(d <= 0 or d > 12 for d in durations):
        warnings.append(f"baseline_runaway_over_12_sec: {durations}")
    return durations, warnings


def train(train_path, valid_path, steps_info):
    import torch
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(max(1, min(4, CPU_THREADS)))
    from peft import get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                              EarlyStoppingCallback, default_data_collator)
    from finetune.train import VieNeuDataset
    from finetune.configs.lora_config import lora_config
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True, dtype=dtype)
    model = get_peft_model(model, lora_config); model.print_trainable_parameters()
    class AuditedVieNeuDataset(VieNeuDataset):
        """Use the official preprocessing but never train on right padding."""
        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            item["labels"] = item["labels"].masked_fill(item["attention_mask"] == 0, -100)
            return item

    train_ds = AuditedVieNeuDataset(str(train_path), tokenizer, max_len=MAX_LEN)
    valid_ds = AuditedVieNeuDataset(str(valid_path), tokenizer, max_len=MAX_LEN)
    args = TrainingArguments(
        output_dir=str(ADAPTER), num_train_epochs=1, learning_rate=1e-5,
        warmup_ratio=0.05, per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=2, max_grad_norm=1.0, logging_steps=10,
        eval_strategy="steps", eval_steps=25, save_strategy="steps", save_steps=25,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=3, report_to="none", dataloader_num_workers=DATA_WORKERS,
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_persistent_workers=DATA_WORKERS > 0,
        remove_unused_columns=False, bf16=torch.cuda.is_available(), fp16=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=valid_ds,
                      data_collator=default_data_collator,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    result = trainer.train()
    trainer.save_model(str(ADAPTER)); tokenizer.save_pretrained(ADAPTER)
    metrics = dict(result.metrics)
    metrics["best_checkpoint"] = trainer.state.best_model_checkpoint
    metrics["train_samples"] = len(train_ds); metrics["valid_samples"] = len(valid_ds)
    return metrics, tokenizer


def merge():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(ADAPTER))
    merged = model.merge_and_unload(); MERGED.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(MERGED, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE, trust_remote_code=True).save_pretrained(MERGED)


def ab_test(test_row):
    from vieneu import Vieneu
    out = RUN / "ab_test"; out.mkdir(parents=True, exist_ok=True)
    result = []
    for label, repo in (("base", BASE), ("merged", MERGED)):
        engine = Vieneu(mode="standard", backbone_repo=str(repo), backbone_device="cuda",
                        codec_repo="neuphonic/neucodec", codec_device="cuda", gguf_filename=None)
        for i, text in enumerate(TEXTS, 1):
            duration = infer(engine, text, Path(test_row["local_path"]), test_row["transcript"], out / f"{label}_{i:02d}.wav")
            result.append({"model": label, "index": i, "text": text, "duration_sec": duration,
                           "warning": "merged_over_2_5x_base_requires_review" if label == "merged" else ""})
        engine.close()
    return result


def main():
    args = parse_args(); RUN.mkdir(parents=True, exist_ok=True)
    rows, test_candidates, train_speakers, valid_speakers = choose_rows(args.seed, args.train_speakers)
    staged = stage(rows); encoded = encode()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    audited = audit(encoded, staged, tokenizer)
    split_paths, train_rows, valid_rows = write_split_metadata(audited, train_speakers, valid_speakers)
    selected_test = None
    baseline_durations = None
    baseline_warnings = []
    for candidate in test_candidates:
        candidate_dir = RUN / "baseline_check" / f"{candidate['speakerID']}__{Path(candidate['filename']).stem}"
        durations, warnings = baseline(candidate, candidate_dir, TEXTS[:3], 1)
        if not warnings:
            extra_durations, extra_warnings = baseline(candidate, candidate_dir, TEXTS[3:], 4)
            durations.extend(extra_durations)
            warnings.extend(extra_warnings)
        if not warnings:
            selected_test = candidate
            baseline_durations = durations
            break
        baseline_warnings.extend([f"{candidate['speakerID']}/{candidate['filename']}: {w}" for w in warnings])
    if selected_test is None:
        report = {
            "status": "blocked_before_training",
            "reason": "No held-out reference passed the baseline checks; LoRA training was not started.",
            "base_model": BASE, "baseline_candidates_tested": len(test_candidates),
            "warnings": baseline_warnings,
        }
        (RUN / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(report["reason"] + " See baseline_check/ and training_report.json.")
    metrics, _ = train(split_paths["train_encoded.csv"], split_paths["valid_encoded.csv"], None)
    merge()
    ab = ab_test(selected_test)
    base_by_index = {r["index"]: r["duration_sec"] for r in ab if r["model"] == "base"}
    ab_failures = [
        r for r in ab if r["model"] == "merged" and
        (r["duration_sec"] > 12 or r["duration_sec"] > 2.5 * base_by_index[r["index"]])
    ]
    report = {
        "base_model": BASE, "test_speaker": selected_test["speakerID"],
        "train_speakers": sorted(train_speakers), "validation_speakers": sorted(valid_speakers),
        "test_reference": selected_test, "baseline_durations_sec": baseline_durations,
        "samples_before_filter": len(audited),
        "train_samples": len(train_rows), "validation_samples": len(valid_rows),
        "max_token_count": max(r["total_token_count"] for r in audited),
        "over_context_samples": sum(r["over_2048"] for r in audited),
        "samples_with_eos": sum(r["eos_in_labels"] for r in audited),
        "epochs_requested": 1, "training_metrics": metrics,
        "baseline_durations_sec": baseline_durations, "ab_test": ab,
        "temperature": TEMPERATURE, "top_k": TOP_K,
        "status": "completed_ab_failed" if ab_failures else "completed",
        "ab_failures": ab_failures,
        "warnings": ["Audio quality and transcript correctness still require listening review."]
    }
    (RUN / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if ab_failures:
        raise RuntimeError("A/B check failed: merged output is runaway or exceeds 2.5x base. See training_report.json.")


if __name__ == "__main__":
    main()
