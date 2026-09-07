"""VieNeu-TTS v2 Nghệ An speaker-held-out LoRA pilot.

The source dataset and the old v3 experiments are read-only.  A small derived
staging dataset is placed below this run's output so the repository's official
v2 encoder and dataset/training code can be reused without rewriting source
data or the v3 pipeline.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "source_code" / "audio_model"
sys.path.insert(0, str(SOURCE_ROOT / "src"))
sys.path.insert(0, str(SOURCE_ROOT))

PREPARED = ROOT / "train" / "output" / "nghean_test2_accent_scale"
MANIFEST = PREPARED / "nghean_eligible_audio_manifest.csv"
RUN = ROOT / "train" / "output" / "nghean_v2_pilot"
PILOT_DATASET = RUN / "pilot_dataset"
ADAPTER = RUN / "adapter"
MERGED = RUN / "merged_model"
BASE = "pnnbao-ump/VieNeu-TTS-0.3B"
TEST_TEXT = "Hôm nay tôi đang thử nghiệm giọng nói tiếng Việt sau khi huấn luyện mô hình."


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--train-speakers", type=int, default=20)
    p.add_argument("--max-samples", type=int, default=80)
    p.add_argument("--seed", type=int, default=37)
    return p.parse_args()


def read_manifest():
    with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("exists") == "True" and r.get("readable") == "True"]


def prepare_staging(seed: int, n_speakers: int, max_samples: int):
    rows = read_manifest()
    # This existing eligible manifest has already passed the repository's
    # audio/QC preparation; its transcript column is named ``transcript``.
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    if not train or not test:
        raise RuntimeError("Expected local clean Nghệ An train and test rows in manifest")

    counts = Counter(r["speakerID"] for r in train)
    selected_speakers = [s for s, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:n_speakers]]
    selected = [r for r in train if r["speakerID"] in selected_speakers]
    random.Random(seed).shuffle(selected)
    selected = selected[:max_samples]
    # Keep the test speaker completely disjoint from the selected train IDs.
    test_speakers = sorted({r["speakerID"] for r in test} - set(selected_speakers))
    if not test_speakers:
        raise RuntimeError("No held-out Nghệ An test speaker remains")
    test_speaker = test_speakers[0]
    test_rows = [r for r in test if r["speakerID"] == test_speaker]

    raw = PILOT_DATASET / "raw_audio"
    raw.mkdir(parents=True, exist_ok=True)
    metadata = []
    for r in selected:
        # Prefix speaker ID; this prevents collisions and records provenance.
        name = f"{r['speakerID']}__{r['filename']}"
        shutil.copy2(r["local_path"], raw / name)
        metadata.append((name, r["transcript"], r["speakerID"]))
    # Official v2 scripts consume metadata.csv / metadata_cleaned.csv.
    for name in ("metadata.csv", "metadata_cleaned.csv"):
        with (PILOT_DATASET / name).open("w", encoding="utf-8", newline="") as f:
            for filename, text, _ in metadata:
                f.write(f"{filename}|{text}\n")
    return selected_speakers, metadata, test_speaker, test_rows[0]


def encode_with_official_script():
    from finetune.data_scripts.encode_data import encode_dataset
    encoded = PILOT_DATASET / "metadata_encoded.csv"
    encode_dataset(dataset_dir=str(PILOT_DATASET), max_samples=2000)
    if not encoded.exists() or not encoded.stat().st_size:
        raise RuntimeError("Official encode_data.py produced no metadata_encoded.csv")
    return encoded


def train_lora(encoded: Path, steps: int):
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator
    from finetune.train import VieNeuDataset
    from finetune.configs.lora_config import lora_config

    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True, dtype=dtype)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Short context keeps this CPU pilot tractable; the official dataset
    # implementation still performs the v2 phonemize/label construction.
    dataset = VieNeuDataset(str(encoded), tokenizer, max_len=512)
    training = TrainingArguments(
        output_dir=str(ADAPTER), max_steps=steps, per_device_train_batch_size=1,
        gradient_accumulation_steps=1, learning_rate=1e-4, warmup_ratio=0.03,
        logging_steps=25, save_strategy="steps", save_steps=100, save_total_limit=3,
        eval_strategy="no", report_to="none", dataloader_num_workers=0,
        bf16=torch.cuda.is_available(), fp16=False, remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(model=model, args=training, train_dataset=dataset,
                      data_collator=default_data_collator)
    result = trainer.train()
    trainer.save_model(str(ADAPTER))
    tokenizer.save_pretrained(ADAPTER)
    return {"train_loss": float(result.training_loss), "global_step": int(result.global_step)}


def merge_lora():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(ADAPTER))
    merged = model.merge_and_unload()
    MERGED.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(MERGED, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE, trust_remote_code=True).save_pretrained(MERGED)
    del merged, model, base
    gc.collect()


def infer(repo: str | Path, ref_audio: Path, ref_text: str, output: Path):
    import torch
    from vieneu import Vieneu

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = Vieneu(mode="standard", backbone_repo=str(repo), backbone_device=device,
                    codec_repo="neuphonic/neucodec", codec_device=device,
                    gguf_filename=None)
    audio = engine.infer(TEST_TEXT, ref_audio=str(ref_audio), ref_text=ref_text,
                         max_chars=256, apply_watermark=False)
    engine.save(audio, str(output))
    duration = len(audio) / engine.sample_rate
    engine.close()
    return float(duration)


def main():
    import soundfile as sf
    a = args()
    if a.steps <= 0 or a.train_speakers <= 0:
        raise ValueError("steps and train-speakers must be positive")
    RUN.mkdir(parents=True, exist_ok=True)
    speakers, selected, test_speaker, test_row = prepare_staging(a.seed, a.train_speakers, a.max_samples)
    encoded = encode_with_official_script()
    metrics = train_lora(encoded, a.steps)
    merge_lora()

    ref_audio = Path(test_row["local_path"])
    merged_duration = infer(MERGED, ref_audio, test_row["transcript"], RUN / "nghean_merged_test.wav")
    base_duration = infer(BASE, ref_audio, test_row["transcript"], RUN / "base_test.wav")
    report = {
        "base_model": BASE, "dataset_source": str(PREPARED),
        "pilot_dataset": str(PILOT_DATASET), "test_speaker": test_speaker,
        "reference_audio": str(ref_audio), "reference_text": test_row["transcript"],
        "test_text": TEST_TEXT, "train_speakers": len(speakers),
        "train_samples": len(selected), "steps": a.steps,
        "lora_rank": 16, "lora_alpha": 32, "learning_rate": 1e-4,
        "checkpoint": str(ADAPTER), "periodic_checkpoints": [str(p) for p in sorted(ADAPTER.glob("checkpoint-*"))],
        "merged_model": str(MERGED), "base_output": str(RUN / "base_test.wav"),
        "merged_output": str(RUN / "nghean_merged_test.wav"),
        "base_duration_sec": float(sf.info(RUN / "base_test.wav").duration),
        "merged_duration_sec": float(sf.info(RUN / "nghean_merged_test.wav").duration),
        "train_metrics": metrics,
    }
    (RUN / "pilot_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
