# Test 5 — Nghệ An regional pilot with one packaged speaker embedding

Test 5 uses the v3 Turbo components already integrated by VieNeu: SEA-G2P for matching text normalization/phonemization, MOSS Audio Tokenizer codes as the acoustic target, and VieNeu's 192D speaker encoder. Training keeps each retained clip's original speaker embedding; inference uses one real in-dataset medoid embedding and no reference WAV or codec codes.

## Data and training safeguards

- Source WAVs and transcripts are left unchanged. Preparation excludes only the four clips reviewed manually for a repeated phrase, incomplete/mismatched transcript, or uncertain word. Generic word-ending filters are avoided because they reject valid Vietnamese phrases such as “triệt để”. `output/nghean_region_no_anchor/quality_audit.json` records each decision.
- Train for 60 steps by default, evaluate and save every 5 steps, then merge the checkpoint with the lowest eval loss. Runs get unique timestamped folders so a longer run cannot overwrite a shorter run or append to its log. The hard limit is 300 steps. Earlier logs favored a checkpoint near step 55, so longer runs can overfit; the script still selects the lowest-eval checkpoint.
- Inference defaults use 0.30 temperature, 1.35 repetition penalty, 180-character chunks, and short pauses (0.05/0.10/0.16 seconds). All chunks use the same packed regional speaker embedding. Long MOSS decoding sends 8-second slices to host RAM to keep VRAM bounded. Runtime context is limited to 2x. VieNeu’s babble guard retries likely repetition; the audio gate checks silence and long internal quiet gaps, while CPU Whisper Small checks each chunk for missing speech, clear phrase loops, and long no-word gaps. After joining, sustained low-energy runs below -47 dBFS that exceed 0.8 seconds are shortened to 0.18 seconds; shorter breaths and pauses remain. The full WAV is checked for transcript coverage and repetition before overwrite. A chunk gets up to 3 retries; a failed full-WAV check leaves the previous demo intact. ASR is a text proxy, not a voice-quality score.
- The latest pause-fix pilot shortened the measured 2.78-second gap after a sentence to 0.84 seconds. The full output is 58.60 seconds; Whisper Small word error rate was 0.18 and maximum word-alignment gap was 1.30 seconds. These are automated measurements, not a guarantee of perfect speech.

The upstream fine-tuning guide requires all clips in one LoRA to be from the same speaker and recommends about 10–30 minutes. This experiment intentionally uses mixed speakers and only about 6 minutes, so it is outside the documented training condition. It remains an experimental multi-speaker LoRA. The 192D speaker embedding carries speaker identity; MOSS codes carry the acoustic realization. The integrated components do not automatically disentangle Nghệ An accent from identity, so one consistent natural speaker and zero generation errors cannot be guaranteed from these data.

## Train and pack

From the repository root, use the project Python environment:

```bash
/home/mediatech/dangtin/trainning/application/conda/envs/tts_5/bin/python train/process/v3/test_5/train_region_no_anchor.py --all --overwrite-stage
```

This stages and prepares the isolated test 5 dataset, excludes the four manually reviewed clips, trains 60 steps, and merges the checkpoint with the lowest validation loss. `--overwrite-stage` replaces only test 5's copied dataset; it does not remove the retained demo or earlier training runs.

## Generate without a reference WAV

```bash
/home/mediatech/dangtin/trainning/application/conda/envs/tts_5/bin/python train/process/v3/test_5/infer_region_no_anchor.py --text "Nội dung cần đọc ở đây."
```

The default output is `output/nghean_region_no_anchor/demo_medoid_trimmed/region_voice_01.wav`; a successful run overwrites that WAV and its report. Set `--output-dir` to compare a candidate elsewhere first.

## Component references

- [VieNeu v3 Turbo inference and speaker embedding](https://github.com/pnnbao97/VieNeu-TTS/blob/main/src/vieneu/_v3_turbo_engine/inference_v3_turbo.py)
- [SEA-G2P](https://github.com/pnnbao97/sea-g2p)
- [MOSS Audio Tokenizer Nano](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer)
