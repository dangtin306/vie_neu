# VieNeu v2 Nghệ An - test_2 (adapter-only + EOS protection)

## Files

train_lora_30_eos.py
test_lora_model.py

## Training design

- Base: pnnbao-ump/VieNeu-TTS-0.3B
- Codec: neuphonic/neucodec
- Exactly 30 SAFE training utterances AFTER official filter + NeuCodec encode +
  token/context/EOS audit.
- PEFT LoRA is saved separately.
- No weight merge.
- <|SPEECH_GENERATION_END|> gets a configurable token-level loss weight.
- Default: 80 epochs, LR 5e-6, EOS weight 5.
- --fast-gpu uses BF16/SDPA when supported. The EOS-weighted loss is computed
  in token windows and gradient checkpointing is enabled so the known ~9.8GB
  card can use batch 2 without the previous CUDA OOM.

## Train

source /root/miniconda3/etc/profile.d/conda.sh
conda activate tts_5
cd /root/media_tech_ai/vie_neu

python train/process/v2/test_2/train_lora_30_eos.py \
  --run-name nghean_v2_lora30_eos \
  --epochs 80 \
  --learning-rate 5e-6 \
  --eos-loss-weight 5 \
  --fast-gpu \
  --overwrite

Nếu dataset đã encode nhưng train bị dừng, chạy lại chỉ phần train:

python train/process/v2/test_2/train_lora_30_eos.py \
  --run-name nghean_v2_lora30_eos \
  --epochs 80 \
  --learning-rate 5e-6 \
  --eos-loss-weight 5 \
  --fast-gpu \
  --train-only

## Expected training output

train/output/nghean_v2_lora30_eos/
├── dataset/
├── adapter/
├── training_report.json
└── valid_eos_audit/

## Runtime test

python train/process/v2/test_2/test_lora_model.py \
  --adapter \
  /root/media_tech_ai/vie_neu/train/output/nghean_v2_lora30_eos/adapter

## Optional BASE vs BASE+LoRA comparison

python train/process/v2/test_2/test_lora_model.py \
  --adapter \
  /root/media_tech_ai/vie_neu/train/output/nghean_v2_lora30_eos/adapter \
  --compare-base

## Custom sentence

python train/process/v2/test_2/test_lora_model.py \
  --adapter \
  /root/media_tech_ai/vie_neu/train/output/nghean_v2_lora30_eos/adapter \
  --text "Hôm nay tôi đang thử một câu mới."

## Inference EOS policy

1. normal: temperature 0.35, top_k 25
2. if fail: low-temp 0.20, top_k 15
3. if both fail: split fragment immediately
4. recurse into smaller phrases/words
5. only at an atomic fragment: alternate sampling, then greedy
6. if still bad: skip that atomic fragment unless --strict

A fragment is accepted only if:

- SPEECH_GENERATION_END was actually generated;
- EOS was not implausibly late;
- decoded duration is within the configurable duration safety budget.

Therefore a missing-EOS fragment is never decoded/saved as runaway audio.
This does NOT mean the model is mathematically guaranteed to emit EOS; it
means the inference pipeline rejects unsafe generations.

## Outputs

train/process/v2/test_2/outputs/
├── lora/
│   ├── lora_01.wav
│   ├── ...
│   └── lora_15.wav
├── base/                 (only with --compare-base)
└── eos_inference_report.json
