"""Runtime-only inspection for the local VieNeu-TTS v3 Turbo engine.

This script intentionally does not train, call backward, create an optimizer,
apply PEFT/LoRA, or modify the upstream source tree.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


TRAIN_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = TRAIN_DIR.parent
SOURCE_AUDIO_MODEL = PROJECT_DIR / "source_code" / "audio_model"
SOURCE_SRC = SOURCE_AUDIO_MODEL / "src"
TEXT_PROCESS_PYTHON = PROJECT_DIR / "source_code" / "text_process" / "python"
CANDIDATES_DIR = PROJECT_DIR / "vimd_hp_candidates"
PREFERRED_CSV = CANDIDATES_DIR / "metadata_preferred_6_15s.csv"
CHECKPOINT = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
MODEL_SUBFOLDER = "update"
MOSS_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
TEST_TEXT = "Hôm nay trời đẹp."


def fail(message: str) -> None:
    raise RuntimeError(message)


def add_local_imports() -> None:
    sys.path.insert(0, str(SOURCE_SRC))
    # Prefer the checked-out text frontend only when its maturin-built Rust
    # extension is present. Otherwise use the installed sea-g2p package in the
    # same environment; it is the same frontend family and keeps this runtime
    # inspection independent of a missing local compiler artifact.
    rust_ext = list((TEXT_PROCESS_PYTHON / "sea_g2p").glob("sea_g2p_rs*.pyd"))
    if rust_ext:
        sys.path.insert(0, str(TEXT_PROCESS_PYTHON))
        print(f"Using local sea-g2p extension: {rust_ext[0]}")
    else:
        print("Local sea-g2p Rust extension not found; using installed sea-g2p package.")


def choose_reference() -> tuple[Path, dict[str, str]]:
    if not PREFERRED_CSV.exists():
        fail(f"Không tìm thấy metadata: {PREFERRED_CSV}")
    with PREFERRED_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"downloaded_file", "duration_sec", "transcript", "speakerID"}
    missing = required - set(rows[0] if rows else ())
    if missing:
        fail(f"CSV thiếu cột: {sorted(missing)}")

    def score(row: dict[str, str]) -> tuple[int, float, str]:
        decision = (row.get("decision") or "").strip().casefold()
        keep_score = 0 if decision == "giữ" else 1
        try:
            duration = float(row.get("duration_sec") or 0)
        except ValueError:
            duration = 999.0
        return keep_score, abs(duration - 9.0), row.get("downloaded_file", "")

    for row in sorted(rows, key=score):
        path = CANDIDATES_DIR / row["downloaded_file"]
        if path.exists():
            return path, row
    fail("Không có WAV ứng viên hợp lệ trong metadata preferred.")


def print_config(engine: Any) -> None:
    cfg = engine.config
    print("\n## 1. Runtime config")
    print(f"model class: {type(engine.model).__module__}.{type(engine.model).__name__}")
    print(f"checkpoint: {CHECKPOINT}")
    print(f"subfolder: {MODEL_SUBFOLDER}")
    print(f"device: {engine.device}")
    print(f"dtype: {engine.dtype}")
    for name in (
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "audio_vocab_size", "n_vq", "audio_pad_token_id",
        "speech_generation_start_token_id", "speech_generation_end_token_id",
        "audio_ref_slot_token_id", "text_vocab_size", "use_speaker_embedding",
    ):
        print(f"{name}: {getattr(cfg, name, '<missing>')}")


def inspect_reference(engine: Any, wav_path: Path, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray | None]:
    import soundfile as sf

    wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    duration = len(wav) / sr if np.ndim(wav) == 1 else wav.shape[0] / sr
    print("\n## 2. Speaker conditioning")
    print(f"filename: {wav_path.name}")
    print(f"duration: {duration:.3f}s")
    print(f"original sample rate: {sr}")
    print(f"speakerID: {row.get('speakerID', '')}")
    print(f"transcript: {row.get('transcript', '')}")
    print(f"use_speaker_embedding: {engine.use_speaker_embedding}")

    spk, ref_codes = engine.prepare_reference(str(wav_path), denoise=False, use_ref_codes=True)
    if spk is not None:
        print(f"speaker_emb shape: {np.asarray(spk).shape}")
        print(f"speaker_emb dtype: {np.asarray(spk).dtype}")
        if engine.model.xvec_proj is not None:
            spk_t = engine._resolve_speaker_emb(spk)
            projected = engine.model.xvec_proj(spk_t.to(engine.dtype))
            print(f"xvec_proj output shape: {tuple(projected.shape)}")
            print(f"speaker anchor added in: _build_inputs_embeds() output, every timestep")
    else:
        print("speaker_emb: None (checkpoint does not enable speaker embedding)")

    print("\n## 3. MOSS encoding")
    if ref_codes is None:
        fail("MOSS không trả ref_codes.")
    ref_codes = np.asarray(ref_codes)
    print(f"MOSS input: mono source is resampled by engine to {engine.SAMPLE_RATE} Hz, then repeated to tokenizer channels")
    print(f"ref_codes shape: {ref_codes.shape}")
    print(f"ref_codes dtype: {ref_codes.dtype}")
    print(f"min code: {int(ref_codes.min())}; max code: {int(ref_codes.max())}")
    print(f"frames/sec: {len(ref_codes) / duration:.3f}")
    print(f"ms/frame: {1000.0 * duration / len(ref_codes):.3f}")
    return np.asarray(spk) if spk is not None else None, ref_codes


def print_text_and_prompt(engine: Any, ref_codes: np.ndarray) -> torch.Tensor:
    from vieneu_utils.phonemize_text import phonemize_text_with_emotions

    phones = phonemize_text_with_emotions(TEST_TEXT)
    token_ids = engine.tokenizer.encode(phones, add_special_tokens=False)
    style_id = engine._resolve_style_id()
    prompt = engine._build_prompt_2d(phones, None, ref_codes, style_id)
    cfg = engine.config
    print("\n## 4. Text/phoneme processing")
    print(f"raw text: {TEST_TEXT}")
    print(f"sea-g2p output: {phones}")
    print(f"token IDs: {token_ids}")
    print("\n## 5. Prompt layout")
    print(f"prompt shape: {tuple(prompt.shape)}")
    t_text = 2 + len(token_ids) + 1
    regions = {
        "STYLE": (0, 1),
        "TEXT_PROMPT_START": (1, 2),
        "PHONEME": (2, t_text - 1),
        "TEXT_PROMPT_END": (t_text - 1, t_text),
        "REFERENCE": (t_text, prompt.shape[0]),
    }
    for name, (start, end) in regions.items():
        print(f"{name}: [{start}:{end}] rows={max(0, end-start)}")
    print(f"reference slot token: {cfg.audio_ref_slot_token_id}")
    print(f"prompt row 0: {prompt[0].tolist()[:min(17, prompt.shape[1])]}")
    print(f"prompt text-end row: {prompt[t_text-1].tolist()[:min(17, prompt.shape[1])]}")
    print(f"prompt first reference row: {prompt[t_text].tolist()[:min(17, prompt.shape[1])]}")
    return prompt


def trace_backbone(engine: Any, prompt: torch.Tensor, speaker_emb: np.ndarray | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    spk_t = engine._resolve_speaker_emb(speaker_emb)
    input_ids = prompt.unsqueeze(0).to(engine.device)
    embeds = engine.model._build_inputs_embeds(input_ids, speaker_emb=spk_t)
    out = engine.model.semantic_backbone(inputs_embeds=embeds, use_cache=True, return_dict=True)
    h0 = out.last_hidden_state[:, -1]
    print("\n## 6. Semantic backbone trace")
    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"inputs_embeds shape: {tuple(embeds.shape)}")
    print(f"semantic output shape: {tuple(out.last_hidden_state.shape)}")
    print(f"global h_0 shape: {tuple(h0.shape)}")
    print(f"past_key_values layers: {len(out.past_key_values)}")
    return h0, out.past_key_values


def trace_frame(engine: Any, h: torch.Tensor, frame_index: int, max_print: int = 16) -> torch.Tensor:
    model = engine.model
    cfg = engine.config
    decoder = model.acoustic_decoder
    local_dtype = next(decoder.parameters()).dtype
    cond = h[0].to(dtype=local_dtype)
    sgs = torch.tensor([cfg.speech_generation_start_token_id], device=engine.device)
    txt = model.text_embeddings(sgs)[0].to(dtype=local_dtype)
    tokens = [cond, txt]
    codes = []
    section = "7. Frame 0 trace" if frame_index == 0 else ("8. Frame 1 trace" if frame_index == 1 else f"Frame {frame_index} trace")
    print(f"\n## {section}")
    print(f"frame_index: {frame_index}")
    for k in range(cfg.n_vq):
        local_in = torch.stack(tokens).unsqueeze(0)
        local_out = decoder(local_in)
        vec = local_out[:, 1] if k == 0 else local_out[:, -1]
        logits = model.audio_lm_heads[k](vec).float()
        code = int(logits.argmax(dim=-1).item())
        codes.append(code)
        print(f"codebook {k}: local input {tuple(local_in.shape)}; local hidden {tuple(local_out.shape)}; logits {tuple(logits.shape)}; selected={code}")
        if k < cfg.n_vq - 1:
            emb = model.audio_embeddings[k](torch.tensor([code], device=engine.device))[0].to(dtype=local_dtype)
            tokens.append(emb)
    return torch.tensor(codes, dtype=torch.long, device=engine.device)


def trace_next_frame(engine: Any, past: Any, h: torch.Tensor, frame_codes: torch.Tensor, speaker_emb: np.ndarray | None) -> tuple[torch.Tensor, Any]:
    cfg = engine.config
    row = torch.full((1, 1, cfg.n_vq + 1), cfg.audio_pad_token_id, dtype=torch.long, device=engine.device)
    row[:, :, 0] = cfg.speech_generation_start_token_id
    row[:, 0, 1:] = frame_codes
    spk_t = engine._resolve_speaker_emb(speaker_emb)
    embeds = engine.model._build_inputs_embeds(row, speaker_emb=spk_t)
    out = engine.model.semantic_backbone(inputs_embeds=embeds, past_key_values=past, use_cache=True, return_dict=True)
    h1 = out.last_hidden_state[:, 0]
    before = past[0][0].shape[2]
    after = before + out.last_hidden_state.shape[1]
    print(f"input/cache length before new frame row: {before}")
    print(f"input/cache length after new frame row: {after} (+1 row with 16 codes)")
    print(f"h_1 shape: {tuple(h1.shape)}")
    return h1, out.past_key_values


def trace_eos(engine: Any, local_h: torch.Tensor, frame_index: int) -> bool:
    # decode_one_frame checks text_lm_head on local slot 0 before appending the next row.
    cfg = engine.config
    with torch.no_grad():
        local_dtype = next(engine.model.acoustic_decoder.parameters()).dtype
        toks = torch.stack([local_h[0].to(local_dtype), engine.model.text_embeddings(torch.tensor([cfg.speech_generation_start_token_id], device=engine.device))[0].to(local_dtype)]).unsqueeze(0)
        local_out = engine.model.acoustic_decoder(toks)
        logits = engine.model.text_lm_head(local_out[0, 0]).float()
        pred = int(logits.argmax().item())
        eos_logit = float(logits[cfg.speech_generation_end_token_id].item())
    print(f"frame {frame_index}: text_lm_head argmax={pred}; speech EOS id={cfg.speech_generation_end_token_id}; EOS logit={eos_logit:.5f}; check occurs after frame append")
    return pred == cfg.speech_generation_end_token_id


def inspect_tying(engine: Any) -> None:
    model = engine.model
    print("\n## 10. Weight tying")
    print(f"text tied: {model.text_lm_head.weight.data_ptr() == model.text_embeddings.weight.data_ptr()}")
    for k, (head, emb) in enumerate(zip(model.audio_lm_heads, model.audio_embeddings)):
        print(f"audio codebook {k} tied: {head.weight.data_ptr() == emb.weight.data_ptr()}")


def inspect_grad_capability(engine: Any, prompt: torch.Tensor, speaker_emb: np.ndarray | None) -> None:
    model = engine.model
    print("\n## 11. Gradient path capability (no backward, no optimizer)")
    model.train(False)
    spk_t = engine._resolve_speaker_emb(speaker_emb)
    ids = prompt.unsqueeze(0).to(engine.device)
    embeds = model._build_inputs_embeds(ids, speaker_emb=spk_t)
    out = model.semantic_backbone(inputs_embeds=embeds, use_cache=False, return_dict=True)
    h = out.last_hidden_state[:, -1]
    local_dtype = next(model.acoustic_decoder.parameters()).dtype
    sgs = torch.tensor([engine.config.speech_generation_start_token_id], device=engine.device)
    local = torch.stack([h[0].to(local_dtype), model.text_embeddings(sgs)[0].to(local_dtype)]).unsqueeze(0)
    local_out = model.acoustic_decoder(local)
    audio_logits = model.audio_lm_heads[0](local_out[:, 1]).float()
    text_logits = model.text_lm_head(local_out[:, 0]).float()
    for name, value in (
        ("semantic_backbone", out.last_hidden_state),
        ("acoustic_decoder", local_out),
        ("audio_lm_heads", audio_logits),
        ("text_lm_head", text_logits),
    ):
        print(f"{name} differentiable: requires_grad={value.requires_grad}; grad_fn={type(value.grad_fn).__name__ if value.grad_fn else None}")
    print("decode_one_frame itself is @torch.no_grad(); direct module calls above retain autograd graph.")


def list_linear_modules(engine: Any) -> None:
    print("\n## 12. Linear modules for later LoRA review")
    for root_name, root in (("semantic_backbone", engine.model.semantic_backbone), ("acoustic_decoder", engine.model.acoustic_decoder)):
        for name, module in root.named_modules():
            if isinstance(module, torch.nn.Linear):
                full = f"{root_name}.{name}" if name else root_name
                print(f"{full}: {module.in_features}->{module.out_features}, params={sum(p.numel() for p in module.parameters())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=2)
    args = parser.parse_args()
    add_local_imports()
    wav_path, row = choose_reference()
    from vieneu._v3_turbo_engine.inference_v3_turbo import VieNeuTTSv3Turbo

    print(f"Loading {CHECKPOINT}/{MODEL_SUBFOLDER} with {MOSS_REPO} ...")
    engine = VieNeuTTSv3Turbo(
        checkpoint_path=CHECKPOINT,
        model_subfolder=MODEL_SUBFOLDER,
        moss_tokenizer_path=MOSS_REPO,
        device="auto",
        dtype="auto",
    )
    print_config(engine)
    speaker_emb, ref_codes = inspect_reference(engine, wav_path, row)
    prompt = print_text_and_prompt(engine, ref_codes)
    h0, past = trace_backbone(engine, prompt, speaker_emb)
    h = h0
    eos = False
    for frame_index in range(max(1, args.max_frames)):
        frame_codes = trace_frame(engine, h, frame_index) if frame_index < 2 else trace_frame(engine, h, frame_index, max_print=0)
        eos = trace_eos(engine, h, frame_index)
        if eos or frame_index + 1 >= max(1, args.max_frames):
            break
        h, past = trace_next_frame(engine, past, h, frame_codes, speaker_emb)
    inspect_tying(engine)
    inspect_grad_capability(engine, prompt, speaker_emb)
    list_linear_modules(engine)
    print("\nInspection completed. No backward, optimizer, LoRA, PEFT, training, or checkpoint was used.")


if __name__ == "__main__":
    main()
