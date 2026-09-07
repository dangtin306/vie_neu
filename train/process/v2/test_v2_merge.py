"""Short VieNeu-TTS v2 LoRA -> merge -> standalone inference smoke test.

This deliberately uses the repository's v2 finetune format and the Standard
engine.  It does not touch the existing v3 server or v3 training artifacts.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
SOURCE = PROJECT / "source_code" / "audio_model"
sys.path.insert(0, str(SOURCE / "src"))
sys.path.insert(0, str(SOURCE))

BASE_MODEL = "pnnbao-ump/VieNeu-TTS-0.3B"
DATASET = PROJECT / "train" / "data" / "dataset_haiphong_test"
OUTPUT = PROJECT / "train" / "output" / "v2_merge_test"
ADAPTER = OUTPUT / "lora_adapter"
MERGED = OUTPUT / "merged_model"
TEXT = "Hôm nay trời đẹp, tôi đang thử nghiệm giọng nói tiếng Việt sau khi huấn luyện."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=10, choices=range(5, 21))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--skip-base", action="store_true")
    return p.parse_args()


def encode_metadata(out_path: Path, max_samples: int = 8):
    """Encode local Hải Phòng samples using the repo's NeuCodec pipeline."""
    import librosa
    import torch
    from neucodec import NeuCodec

    source_meta = DATASET / "metadata_cleaned.csv"
    if not source_meta.exists():
        source_meta = DATASET / "metadata.csv"
    rows = []
    with source_meta.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="|"):
            if len(row) >= 2 and (DATASET / "raw_audio" / row[0]).exists():
                rows.append((row[0], row[1]))
    if not rows:
        raise RuntimeError(f"No usable local samples found in {source_meta}")
    random.shuffle(rows)
    rows = rows[:max_samples]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(device).eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for filename, text in rows:
            wav, _ = librosa.load(DATASET / "raw_audio" / filename, sr=16000, mono=True)
            audio = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.inference_mode():
                codes = codec.encode_code(audio).squeeze().detach().cpu().numpy().reshape(-1).tolist()
            if not codes:
                continue
            out.write(f"{filename}|{text}|{json.dumps([int(x) for x in codes])}\n")
    del codec
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not out_path.stat().st_size:
        raise RuntimeError("NeuCodec produced no encoded samples")
    return out_path


def train_lora(encoded: Path, steps: int):
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator
    from vieneu_utils.phonemize_text import phonemize_with_dict
    from finetune.configs.lora_config import lora_config

    class Dataset(torch.utils.data.Dataset):
        def __init__(self):
            self.items = []
            for line in encoded.read_text(encoding="utf-8").splitlines():
                filename, text, codes = line.split("|", 2)
                self.items.append((text, json.loads(codes)))
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            text, codes = self.items[i]
            phones = phonemize_with_dict(text)
            code_text = "".join(f"<|speech_{x}|>" for x in codes)
            prompt = f"<|TEXT_PROMPT_START|>{phones}<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>{code_text}<|SPEECH_GENERATION_END|>"
            ids = self.tokenizer.encode(prompt)[:2048]
            input_ids = torch.tensor(ids, dtype=torch.long)
            labels = torch.full_like(input_ids, -100)
            start = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
            positions = (input_ids == start).nonzero(as_tuple=True)[0]
            if len(positions):
                labels[positions[0]:] = input_ids[positions[0]:]
            return {"input_ids": input_ids, "labels": labels, "attention_mask": torch.ones_like(input_ids)}

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True, torch_dtype=dtype)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    args = TrainingArguments(
        output_dir=str(ADAPTER), max_steps=steps, per_device_train_batch_size=1,
        gradient_accumulation_steps=1, learning_rate=2e-4, warmup_ratio=0.1,
        logging_steps=1, save_strategy="no", eval_strategy="no", report_to="none",
        bf16=torch.cuda.is_available(), fp16=False, dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    Trainer(model=model, args=args, train_dataset=Dataset(), data_collator=default_data_collator).train()
    model.save_pretrained(ADAPTER)
    tokenizer.save_pretrained(ADAPTER)
    return tokenizer


def merge_model():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, ADAPTER)
    merged = model.merge_and_unload()
    MERGED.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(MERGED, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True).save_pretrained(MERGED)
    del merged, model, base
    gc.collect()


def infer_standalone(repo: str | Path, ref_audio: Path, ref_text: str, output: Path):
    import torch
    from vieneu import Vieneu

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # gguf_filename=None is required here: load the v2 Transformers model,
    # never the default v3 Turbo/GGUF path.
    engine = Vieneu(mode="standard", backbone_repo=str(repo), backbone_device=device,
                    codec_repo="neuphonic/neucodec", codec_device=device,
                    gguf_filename=None)
    audio = engine.infer(TEXT, ref_audio=str(ref_audio), ref_text=ref_text,
                         max_chars=256, apply_watermark=False)
    engine.save(audio, str(output))
    engine.close()


def main():
    args = parse_args()
    random.seed(args.seed)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    encoded = OUTPUT / "metadata_encoded.csv"
    print(f"[v2] dataset: {DATASET}")
    encode_metadata(encoded)
    rows = list(csv.reader((DATASET / "metadata_cleaned.csv").open(encoding="utf-8"), delimiter="|"))
    ref_name, ref_text = next(r[0:2] for r in rows if len(r) >= 2 and (DATASET / "raw_audio" / r[0]).exists())
    ref_audio = DATASET / "raw_audio" / ref_name
    print(f"[v2] training LoRA for {args.steps} steps")
    train_lora(encoded, args.steps)
    print("[v2] merging with PeftModel.merge_and_unload()")
    merge_model()
    print("[v2] loading merged model independently")
    infer_standalone(MERGED, ref_audio, ref_text, OUTPUT / "merged_test.wav")
    if not args.skip_base:
        print("[v2] generating base comparison")
        infer_standalone(BASE_MODEL, ref_audio, ref_text, OUTPUT / "base_test.wav")
    print(f"[v2] complete: {OUTPUT / 'merged_test.wav'}")


if __name__ == "__main__":
    main()
