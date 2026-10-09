# Test 4: Nghệ An regional LoRA

Experimental shared LoRA trained from multiple Nghệ An speakers. VieNeu documents one speaker per LoRA; this multi-speaker setup is an experiment, not an upstream-supported recipe.

## Training data

- Original staged clips: `output/nghean_region_one_lora/dataset/` (preserved).
- Cleaned training copy: `output/nghean_region_one_lora/dataset_training_smooth_v2/` (24 clips, 6.0 minutes).
- Excluded only from this training copy: `0015_37_0062.wav`, `0017_37_0077.wav`, `0019_37_0082.wav` (spoken repeats absent from transcript) and `0024_37_0123.wav` (audio was unrelated to its transcript).
- `prepare_dataset.py` creates a separate speaker embedding for each remaining clip; all rows update the same LoRA.
- Silero VAD shortens detected non-speech gaps longer than 0.42 seconds to 0.18 seconds. Source WAVs remain unchanged.

## Current model

`output/nghean_region_one_lora/training/nghean_region_one_lora_smooth_v2_90/merged/`

90 steps, rank 16, alpha 32, dropout 0.05, backbone target, learning rate `0.0002`. Validation loss: 4.8020; `acc_cb0`: 0.166. The merged model is the only retained model; intermediate checkpoints and the separate adapter were removed. Training arguments and logs remain beside it.

## Current listening sample

`output/nghean_region_one_lora/demo/region_voice_01.wav` (56.31 seconds), generated with reference `0008_37_0006.wav`. Whisper did not flag an exact repeated phrase in its transcript; listen to the WAV for the final quality judgment.

Use `infer_region_one_voice.py` with an explicit `--merged` path above; its default model path is from an older run.
