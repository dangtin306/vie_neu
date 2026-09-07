"""Small, isolated VieNeu-TTS LoRA pipeline test for ViMD HaiPhong data.

This script intentionally does not modify the VieNeu-TTS source tree.  It reuses
the official filter, NeuCodec encoder, phonemizer, dataset and LoRA config while
writing all generated data and checkpoints below this ``train`` directory.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


SCRIPT_DIR = Path(__file__).resolve().parent.parent
VIE_NEU_DIR = SCRIPT_DIR.parent
SOURCE_REPO = VIE_NEU_DIR / "source_code" / "audio_model"
SOURCE_SRC = SOURCE_REPO / "src"
SOURCE_FINETUNE = SOURCE_REPO / "finetune"
SOURCE_CANDIDATES = VIE_NEU_DIR / "vimd_hp_candidates"

DATASET_DIR = SCRIPT_DIR / "data" / "dataset_haiphong_test"
RAW_AUDIO_DIR = DATASET_DIR / "raw_audio"
OUTPUT_DIR = SCRIPT_DIR / "output" / "haiphong_test"
BASE_MODEL = "pnnbao-ump/VieNeu-TTS-0.3B"
TRAIN_STEPS = 30


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_official_module(name: str, path: Path) -> Any:
    if not path.exists():
        fail(f"Không tìm thấy source chính thức: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        fail(f"Không thể load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_source_rows() -> List[Dict[str, str]]:
    preferred = SOURCE_CANDIDATES / "metadata_preferred_6_15s.csv"
    if not preferred.exists():
        fail(f"Không tìm thấy CSV nguồn: {preferred}")

    with preferred.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {
            "downloaded_file",
            "transcript",
            "speakerID",
            "duration_sec",
            "accent_rating",
            "noise_rating",
            "style_rating",
            "decision",
            "notes",
        }
        missing = sorted(required - set(fields))
        if missing:
            fail(f"CSV thiếu cột đã inspect: {', '.join(missing)}")
        rows = list(reader)

    decisions = {row["decision"].strip().casefold() for row in rows if row["decision"].strip()}
    if decisions:
        rows = [row for row in rows if row["decision"].strip().casefold() == "giữ"]

    selected: List[Dict[str, str]] = []
    for row in rows:
        filename = row["downloaded_file"].strip()
        transcript = row["transcript"].strip()
        if not filename or not transcript:
            continue
        if "|" in filename or "|" in transcript:
            continue
        source_audio = (SOURCE_CANDIDATES / filename).resolve()
        if SOURCE_CANDIDATES.resolve() not in source_audio.parents:
            continue
        if not source_audio.is_file():
            continue
        selected.append(row)
    return selected


def prepare_metadata(rows: Iterable[Dict[str, str]]) -> int:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = DATASET_DIR / "metadata.csv"

    count = 0
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            filename = row["downloaded_file"].strip()
            source_audio = SOURCE_CANDIDATES / filename
            target_audio = RAW_AUDIO_DIR / filename
            if not target_audio.exists():
                try:
                    os.link(source_audio, target_audio)
                except OSError:
                    shutil.copy2(source_audio, target_audio)
            handle.write(f"{filename}|{row['transcript'].strip()}\n")
            count += 1
    return count


def count_metadata_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8") .splitlines() if line.strip())


def metadata_stats(encoded_path: Path, source_rows: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    total_duration = 0.0
    speakers: Dict[str, List[float]] = defaultdict(list)
    encoded_count = 0
    if encoded_path.exists():
        with encoded_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("|", 2)
                if len(parts) < 3:
                    continue
                row = source_rows.get(parts[0])
                if row is None:
                    continue
                duration = float(row["duration_sec"])
                speaker = row["speakerID"]
                encoded_count += 1
                total_duration += duration
                speakers[speaker].append(duration)
    return {
        "encoded_count": encoded_count,
        "total_duration": total_duration,
        "speakers": speakers,
    }


def print_stats(input_count: int, filtered_count: int, encoded_path: Path, rows: List[Dict[str, str]]) -> None:
    source_rows = {row["downloaded_file"]: row for row in rows}
    stats = metadata_stats(encoded_path, source_rows)
    print(f"Samples đầu vào: {input_count}")
    print(f"Samples qua filter: {filtered_count}")
    print(f"Samples encode thành công: {stats['encoded_count']}")
    print(f"Tổng duration encoded: {stats['total_duration']:.2f}s ({stats['total_duration'] / 60:.2f} phút)")
    print(f"Số speaker: {len(stats['speakers'])}")
    print("Duration theo speaker:")
    for speaker in sorted(stats["speakers"]):
        durations = stats["speakers"][speaker]
        print(f"  {speaker}: {len(durations)} samples, {sum(durations):.2f}s")
    if stats["encoded_count"] == 0:
        fail("Encode thành công bằng 0; dừng, không chuyển sang training.")


def prepare() -> None:
    rows = read_source_rows()
    input_count = len(rows)
    prepare_metadata(rows)
    print(f"Đã tạo metadata nguồn với {input_count} samples.")

    filter_module = load_official_module("official_filter_data", SOURCE_FINETUNE / "data_scripts" / "filter_data.py")
    filter_module.filter_and_process_dataset(dataset_dir=str(DATASET_DIR))
    cleaned_path = DATASET_DIR / "metadata_cleaned.csv"
    filtered_count = count_metadata_lines(cleaned_path)

    encode_module = load_official_module("official_encode_data", SOURCE_FINETUNE / "data_scripts" / "encode_data.py")
    try:
        encode_module.encode_dataset(dataset_dir=str(DATASET_DIR), max_samples=max(input_count, 1))
    except Exception as exc:
        message = str(exc)
        if "gated" in message.lower() or "403" in message or "neuphonic/neucodec" in message:
            fail(
                "NeuCodec bị Hugging Face từ chối (gated model). "
                "`hf auth login` đã xác thực token nhưng chưa cấp quyền model. "
                "Hãy mở https://huggingface.co/neuphonic/neucodec, đăng nhập "
                "bằng đúng tài khoản, bấm Agree and access/chấp nhận điều kiện "
                "chia sẻ contact information; nếu được yêu cầu thì chờ Hugging Face "
                "duyệt quyền. Sau đó chạy lại `hf auth login` trong env tts_5 và "
                "thử lại --prepare. "
                f"Chi tiết gốc: {exc}"
            )
        raise
    encoded_path = DATASET_DIR / "metadata_encoded.csv"

    # Validate text through the same phonemizer used by official train.py.
    sys.path.insert(0, str(SOURCE_SRC))
    sys.path.insert(0, str(SOURCE_REPO))
    from vieneu_utils.phonemize_text import phonemize_with_dict

    phoneme_ok = 0
    with cleaned_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|", 1)
            if len(parts) == 2:
                phonemize_with_dict(parts[1])
                phoneme_ok += 1
    print(f"Phonemize thành công: {phoneme_ok}")
    print_stats(input_count, filtered_count, encoded_path, rows)
    print(f"Dataset test: {DATASET_DIR}")


def get_device_precision(torch: Any) -> tuple[str, bool, bool]:
    if not torch.cuda.is_available():
        return "cpu", False, False
    bf16 = bool(torch.cuda.is_bf16_supported())
    return "cuda", bf16, not bf16


def train() -> None:
    encoded_path = DATASET_DIR / "metadata_encoded.csv"
    if not encoded_path.exists():
        fail(f"Chưa có {encoded_path}; chạy --prepare trước.")

    sys.path.insert(0, str(SOURCE_SRC))
    sys.path.insert(0, str(SOURCE_REPO))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator
    from peft import get_peft_model

    official_train = load_official_module("official_train", SOURCE_FINETUNE / "train.py")
    official_config = load_official_module("official_lora_config", SOURCE_FINETUNE / "configs" / "lora_config.py")
    device, use_bf16, use_fp16 = get_device_precision(torch)
    if device == "cpu":
        print("CUDA không khả dụng; chỉ prepare dataset, không chạy training CPU.")
        return

    precision = "BF16" if use_bf16 else "FP16"
    print("--- Training summary ---")
    print("Province: HaiPhong")
    print("Source: ViMD")
    print(f"Samples: {count_metadata_lines(encoded_path)}")
    rows = read_source_rows()
    print(f"Speakers: {len({row['speakerID'] for row in rows})}")
    print(f"Duration: {sum(float(row['duration_sec']) for row in rows):.2f}s")
    print(f"Device: {device}")
    print(f"Precision: {precision}")
    print(f"Base model: {BASE_MODEL}")
    print(f"Train steps: {TRAIN_STEPS}")
    print(f"Output: {OUTPUT_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype, device_map="auto")
    dataset = official_train.VieNeuDataset(str(encoded_path), tokenizer)
    model = get_peft_model(model, official_config.lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        do_train=True,
        do_eval=False,
        max_steps=TRAIN_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=official_config.training_config["learning_rate"],
        warmup_ratio=official_config.training_config["warmup_ratio"],
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=1,
        save_steps=TRAIN_STEPS,
        eval_strategy="no",
        save_strategy="steps",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=default_data_collator,
    )
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Đã lưu LoRA tại: {OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--train", action="store_true")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.train:
        train()
    else:
        prepare()
        train()


if __name__ == "__main__":
    main()
