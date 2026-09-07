"""Perturbation + FP32 verification for v3 Turbo EOS slot behavior.

For fixed h_t, compares text_lm_head(slot 0) after:
  A: true codebooks 0..15
  B: random codebooks 0..15
  C: random codebook 0 only
  D: random codebook 15 only

No training, backward, optimizer, LoRA, or checkpoint is used.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


TRAIN_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = TRAIN_DIR.parent
SOURCE_SRC = PROJECT_DIR / "source_code" / "audio_model" / "src"
TEXT_PROCESS_PYTHON = PROJECT_DIR / "source_code" / "text_process" / "python"
sys.path.insert(0, str(SOURCE_SRC))
if list((TEXT_PROCESS_PYTHON / "sea_g2p").glob("sea_g2p_rs*.pyd")):
    sys.path.insert(0, str(TEXT_PROCESS_PYTHON))

from eos_placement_test import CANDIDATES_DIR, MOSS_REPO, CHECKPOINT, MODEL_SUBFOLDER, select_rows


def load_engine():
    from vieneu._v3_turbo_engine.inference_v3_turbo import VieNeuTTSv3Turbo

    return VieNeuTTSv3Turbo(
        checkpoint_path=CHECKPOINT,
        model_subfolder=MODEL_SUBFOLDER,
        moss_tokenizer_path=MOSS_REPO,
        device="auto",
        dtype="auto",
    )


def build_prompt(engine: Any, text: str, codes: np.ndarray) -> torch.Tensor:
    from vieneu_utils.phonemize_text import phonemize_text_with_emotions

    phones = phonemize_text_with_emotions(text)
    return engine._build_prompt_2d(phones, None, codes, engine._resolve_style_id())


def prefill(engine: Any, prompt: torch.Tensor, speaker_emb: np.ndarray | None):
    model = engine.model
    ids = prompt.unsqueeze(0).to(engine.device)
    spk = engine._resolve_speaker_emb(speaker_emb)
    out = model.semantic_backbone(inputs_embeds=model._build_inputs_embeds(ids, speaker_emb=spk), use_cache=True, return_dict=True)
    return out.last_hidden_state[:, -1], out.past_key_values


def append_frame(engine: Any, past: Any, frame: torch.Tensor, speaker_emb: np.ndarray | None):
    cfg = engine.config
    model = engine.model
    row = torch.full((1, 1, cfg.n_vq + 1), cfg.audio_pad_token_id, dtype=torch.long, device=engine.device)
    row[:, :, 0] = cfg.speech_generation_start_token_id
    row[:, 0, 1:] = frame
    spk = engine._resolve_speaker_emb(speaker_emb)
    out = model.semantic_backbone(
        inputs_embeds=model._build_inputs_embeds(row, speaker_emb=spk),
        past_key_values=past,
        use_cache=True,
        return_dict=True,
    )
    return out.last_hidden_state[:, 0], out.past_key_values


def logits_for_codes(engine: Any, h: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    model = engine.model
    cfg = engine.config
    local_dtype = next(model.acoustic_decoder.parameters()).dtype
    cond = h[0].to(dtype=local_dtype)
    start_id = torch.tensor([cfg.speech_generation_start_token_id], device=engine.device)
    start = model.text_embeddings(start_id)[0].to(dtype=local_dtype)
    tokens = [cond, start]
    for k in range(cfg.n_vq):
        current = torch.stack(tokens).unsqueeze(0)
        if k < cfg.n_vq - 1:
            true_code = codes[k].view(1)
            tokens.append(model.audio_embeddings[k](true_code)[0].to(dtype=local_dtype))
    out = model.acoustic_decoder(current)
    return model.text_lm_head(out[0, 0]).float()


def measure(engine: Any, h: torch.Tensor, true_codes: torch.Tensor, seed: int) -> dict[str, float]:
    cfg = engine.config
    generator = torch.Generator(device=engine.device).manual_seed(seed)
    random_codes = torch.randint(0, cfg.audio_vocab_size, true_codes.shape, generator=generator, device=engine.device)
    cases = {
        "A_true": true_codes,
        "B_random_all": random_codes,
        "C_random_code0": true_codes.clone(),
        "D_random_code15": true_codes.clone(),
    }
    cases["C_random_code0"][0] = random_codes[0]
    cases["D_random_code15"][15] = random_codes[15]
    base = logits_for_codes(engine, h, cases["A_true"])
    out: dict[str, float] = {}
    for name, codes in cases.items():
        logits = logits_for_codes(engine, h, codes)
        out[f"{name}_delta"] = float((logits - base).abs().max().cpu())
        out[f"{name}_delta_5"] = float((logits[5] - base[5]).abs().cpu())
        out[f"{name}_delta_6"] = float((logits[6] - base[6]).abs().cpu())
    return out


def run_precision(engine: Any, rows: list[dict[str, str]], precision: str) -> None:
    if precision == "fp32":
        engine.model.float()
    engine.model.eval()
    print(f"\n## Precision: {precision}; model dtype: {next(engine.model.parameters()).dtype}")
    for index, row in enumerate(rows, 1):
        path = CANDIDATES_DIR / row["downloaded_file"]
        text = (row.get("transcript") or "").strip()
        with torch.no_grad():
            speaker_emb, codes_np = engine.prepare_reference(str(path), denoise=False, use_ref_codes=True)
            codes = torch.as_tensor(np.asarray(codes_np), dtype=torch.long, device=engine.device)
            prompt = build_prompt(engine, text, np.asarray(codes_np))
            h0, past = prefill(engine, prompt, speaker_emb)
            h_t = h0
            for t in range(codes.shape[0]):
                if t + 1 < codes.shape[0]:
                    h_t, past = append_frame(engine, past, codes[t], speaker_emb)
            # Feed the final true frame too: this is genuinely h_T, after
            # prompt + frame_0 ... frame_{T-1}.
            h_t, past = append_frame(engine, past, codes[-1], speaker_emb)
            results_0 = measure(engine, h0, codes[0], seed=1000 + index)
            results_t = measure(engine, h_t, codes[-1], seed=2000 + index)
        print(f"\n[{index}/{len(rows)}] {path.name} speaker={row.get('speakerID', '')} frames={codes.shape[0]}")
        for label, values in (("h0", results_0), ("hT", results_t)):
            print(f"  {label}: " + ", ".join(f"{key}={value:.8f}" for key, value in values.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=int, default=3)
    args = parser.parse_args()
    rows = select_rows(max(1, args.clips))
    engine = load_engine()
    print(f"Checkpoint: {CHECKPOINT}/{MODEL_SUBFOLDER}; clips={len(rows)}")
    print("A=true, B=random all, C=random codebook 0, D=random codebook 15")
    print("Delta is max absolute difference from A; no_grad only.")
    run_precision(engine, rows, "bf16")
    run_precision(engine, rows, "fp32")
    print("\nNo backward(), optimizer.step(), LoRA/PEFT, weight update, or checkpoint save was performed.")


if __name__ == "__main__":
    main()
