"""Prototype audio teacher-forced loss for local VieNeu-TTS v3 Turbo.

This is an objective/shape test only. It deliberately does not call backward(),
create an optimizer, apply PEFT/LoRA, update weights, or save a checkpoint.
It uses the true MOSS code at every teacher-forcing position.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


TRAIN_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = TRAIN_DIR.parent
SOURCE_SRC = PROJECT_DIR / "source_code" / "audio_model" / "src"
TEXT_PROCESS_PYTHON = PROJECT_DIR / "source_code" / "text_process" / "python"
CANDIDATES_DIR = PROJECT_DIR / "vimd_hp_candidates"
PREFERRED_CSV = CANDIDATES_DIR / "metadata_preferred_6_15s.csv"
CHECKPOINT = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
MODEL_SUBFOLDER = "update"
MOSS_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"


def add_imports() -> None:
    sys.path.insert(0, str(SOURCE_SRC))
    # Use the checked-out frontend only if its compiled Rust extension exists.
    # Otherwise the same environment's installed sea-g2p package is used.
    if list((TEXT_PROCESS_PYTHON / "sea_g2p").glob("sea_g2p_rs*.pyd")):
        sys.path.insert(0, str(TEXT_PROCESS_PYTHON))


def choose_sample() -> tuple[Path, dict[str, str]]:
    with PREFERRED_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"CSV rỗng: {PREFERRED_CSV}")

    def key(row: dict[str, str]) -> tuple[int, float, str]:
        decision = (row.get("decision") or "").strip().casefold()
        try:
            duration = float(row.get("duration_sec") or 0)
        except ValueError:
            duration = 999.0
        return (0 if decision == "giữ" else 1, abs(duration - 9.0), row.get("downloaded_file", ""))

    for row in sorted(rows, key=key):
        path = CANDIDATES_DIR / row["downloaded_file"]
        if path.exists():
            return path, row
    raise RuntimeError("Không tìm thấy WAV hợp lệ từ metadata preferred.")


def cache_length(past: Any) -> int:
    # Transformers versions used by Qwen may return a tuple cache or DynamicCache.
    if hasattr(past, "get_seq_length"):
        return int(past.get_seq_length())
    return int(past[0][0].shape[2])


def load_engine():
    from vieneu._v3_turbo_engine.inference_v3_turbo import VieNeuTTSv3Turbo

    return VieNeuTTSv3Turbo(
        checkpoint_path=CHECKPOINT,
        model_subfolder=MODEL_SUBFOLDER,
        moss_tokenizer_path=MOSS_REPO,
        device="auto",
        dtype="auto",
    )


def make_prompt(engine: Any, text: str, ref_codes: np.ndarray) -> torch.Tensor:
    from vieneu_utils.phonemize_text import phonemize_text_with_emotions

    phones = phonemize_text_with_emotions(text)
    prompt = engine._build_prompt_2d(
        phones,
        None,
        ref_codes,
        engine._resolve_style_id(),
    )
    print(f"text: {text}")
    print(f"phonemes: {phones}")
    print(f"prompt shape: {tuple(prompt.shape)}")
    return prompt


def semantic_prefill(engine: Any, prompt: torch.Tensor, speaker_emb: np.ndarray | None):
    model = engine.model
    ids = prompt.unsqueeze(0).to(engine.device)
    spk = engine._resolve_speaker_emb(speaker_emb)
    embeds = model._build_inputs_embeds(ids, speaker_emb=spk)
    out = model.semantic_backbone(
        inputs_embeds=embeds,
        use_cache=True,
        return_dict=True,
    )
    return out.last_hidden_state[:, -1], out.past_key_values


def append_true_frame(
    engine: Any,
    past: Any,
    true_frame: torch.Tensor,
    speaker_emb: np.ndarray | None,
):
    cfg = engine.config
    model = engine.model
    row = torch.full(
        (1, 1, cfg.n_vq + 1),
        cfg.audio_pad_token_id,
        dtype=torch.long,
        device=engine.device,
    )
    row[:, :, 0] = cfg.speech_generation_start_token_id
    row[:, 0, 1:] = true_frame
    spk = engine._resolve_speaker_emb(speaker_emb)
    embeds = model._build_inputs_embeds(row, speaker_emb=spk)
    out = model.semantic_backbone(
        inputs_embeds=embeds,
        past_key_values=past,
        use_cache=True,
        return_dict=True,
    )
    # With a KV cache the output contains only the newly supplied row.
    return out.last_hidden_state[:, 0], out.past_key_values


def teacher_force_frame(engine: Any, h: torch.Tensor, target: torch.Tensor):
    """Return 16 logits, using true previous codebook IDs within the frame."""
    model = engine.model
    cfg = engine.config
    local_dtype = next(model.acoustic_decoder.parameters()).dtype
    cond = h[0].to(dtype=local_dtype)
    sgs = torch.tensor([cfg.speech_generation_start_token_id], device=engine.device)
    speech_start = model.text_embeddings(sgs)[0].to(dtype=local_dtype)
    local_tokens = [cond, speech_start]
    logits_by_codebook = []

    for k in range(cfg.n_vq):
        local_input = torch.stack(local_tokens).unsqueeze(0)
        local_out = model.acoustic_decoder(local_input)
        # This matches decode_one_frame(): codebook 0 uses slot 1; later
        # codebooks use the newly appended final local slot.
        hidden = local_out[:, 1] if k == 0 else local_out[:, -1]
        logits = model.audio_lm_heads[k](hidden).float()
        logits_by_codebook.append(logits)

        if k + 1 < cfg.n_vq:
            true_code = target[k].view(1)
            true_emb = model.audio_embeddings[k](true_code)[0].to(dtype=local_dtype)
            local_tokens.append(true_emb)

    return logits_by_codebook


def frame_loss(logits_by_codebook, target: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
    losses = [
        F.cross_entropy(logits, target[k].view(1))
        for k, logits in enumerate(logits_by_codebook)
    ]
    stacked = torch.stack(losses)
    return stacked.mean(), [float(x.detach().cpu()) for x in losses]


def main() -> None:
    parser = argparse.ArgumentParser(description="One-sample v3 Turbo teacher-forced MOSS CE test")
    parser.add_argument("--max-frames", type=int, default=1, help="1 = TEST 1; increase for TEST 2")
    args = parser.parse_args()
    if args.max_frames < 1:
        raise SystemExit("--max-frames phải >= 1")

    add_imports()
    wav_path, row = choose_sample()
    engine = load_engine()
    model = engine.model
    model.eval()

    # Reference and target deliberately use the same real utterance for this
    # first objective test. No denoising is introduced into the target path.
    speaker_emb, reference_codes = engine.prepare_reference(
        str(wav_path), denoise=False, use_ref_codes=True
    )
    target_wav, target_sr = engine._load_mono(str(wav_path), None)
    target_codes = engine._encode_ref_wav(target_wav, target_sr)
    target_codes_t = torch.as_tensor(target_codes, dtype=torch.long, device=engine.device)

    text = row.get("transcript", "").strip()
    if not text:
        raise RuntimeError("Sample không có transcript; không được tự viết lại text.")
    prompt = make_prompt(engine, text, np.asarray(reference_codes))

    print("\n## Teacher-forced audio loss")
    print(f"sample: {wav_path.name}")
    print(f"speakerID: {row.get('speakerID', '')}")
    print(f"reference codes: {tuple(np.asarray(reference_codes).shape)}")
    print(f"target codes: {tuple(target_codes_t.shape)}")
    print(f"testing frames: {min(args.max_frames, target_codes_t.shape[0])}")
    print("teacher forcing: TRUE MOSS code for previous frame and previous codebook")
    print("EOS loss: DISABLED")
    print("backward/optimizer/LoRA: DISABLED")

    # Gradient tracking stays enabled. We inspect requires_grad/grad_fn and do
    # not call backward(), so no weights or .grad buffers are updated.
    h, past = semantic_prefill(engine, prompt, speaker_emb)
    frame_losses = []
    per_codebook = []
    for t in range(min(args.max_frames, target_codes_t.shape[0])):
        logits = teacher_force_frame(engine, h, target_codes_t[t])
        loss_t, losses_k = frame_loss(logits, target_codes_t[t])
        if not torch.isfinite(loss_t).item():
            raise RuntimeError(f"Nem audio loss tại frame {t}: {loss_t}")
        frame_losses.append(loss_t)
        per_codebook.append(losses_k)
        print(
            f"frame {t}: loss={float(loss_t.detach().cpu()):.6f}; "
            f"logits={[tuple(x.shape) for x in logits]}; "
            f"requires_grad={loss_t.requires_grad}; "
            f"grad_fn={type(loss_t.grad_fn).__name__ if loss_t.grad_fn else None}"
        )
        if t + 1 < min(args.max_frames, target_codes_t.shape[0]):
            h, past = append_true_frame(engine, past, target_codes_t[t], speaker_emb)
            print(f"  next-frame cache length: {cache_length(past)}; h shape: {tuple(h.shape)}")

    total_loss = torch.stack(frame_losses).mean()
    mean_by_codebook = np.asarray(per_codebook, dtype=np.float64).mean(axis=0)
    print("\n## Result")
    print(f"L_audio: {float(total_loss.detach().cpu()):.6f}")
    print(f"L_audio finite: {bool(torch.isfinite(total_loss).item())}")
    print(f"L_audio requires_grad: {total_loss.requires_grad}")
    print(f"L_audio grad_fn: {type(total_loss.grad_fn).__name__ if total_loss.grad_fn else None}")
    print("mean CE by codebook:")
    for k, value in enumerate(mean_by_codebook):
        print(f"  codebook {k}: {value:.6f}")
    print("No backward(), optimizer.step(), LoRA/PEFT, weight update, or checkpoint save was performed.")


if __name__ == "__main__":
    main()
