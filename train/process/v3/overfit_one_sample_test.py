"""TEST 4A: one-sample audio-loss overfit for local VieNeu-TTS v3 Turbo.

This deliberately trains only acoustic_decoder.layers.0. The semantic backbone,
MOSS tokenizer, speaker encoder, embeddings, audio heads, and text head remain
frozen. No EOS loss is used. The script is an experiment artifact only and does
not modify upstream source or save a checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
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
REFERENCE_FILE = "candidate_042.wav"
CHECKPOINT = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
MODEL_SUBFOLDER = "update"
MOSS_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
CHECKPOINTS = (0, 1, 5, 10, 20, 50, 100)

sys.path.insert(0, str(SOURCE_SRC))
if list((TEXT_PROCESS_PYTHON / "sea_g2p").glob("sea_g2p_rs*.pyd")):
    sys.path.insert(0, str(TEXT_PROCESS_PYTHON))


def load_row() -> tuple[Path, dict[str, str]]:
    path = CANDIDATES_DIR / REFERENCE_FILE
    with PREFERRED_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("downloaded_file") == REFERENCE_FILE and path.exists():
            return path, row
    raise RuntimeError(f"Không tìm thấy sample cố định {path} trong metadata preferred.")


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


def precompute_h(engine: Any, prompt: torch.Tensor, codes: torch.Tensor, speaker_emb: np.ndarray | None):
    """Compute h_0..h_{T-1} with frozen semantic backbone and true history."""
    model = engine.model
    model.semantic_backbone.eval()
    with torch.no_grad():
        ids = prompt.unsqueeze(0).to(engine.device)
        spk = engine._resolve_speaker_emb(speaker_emb)
        embeds = model._build_inputs_embeds(ids, speaker_emb=spk)
        semantic_dtype = next(model.semantic_backbone.parameters()).dtype
        out = model.semantic_backbone(
            inputs_embeds=embeds.to(dtype=semantic_dtype),
            use_cache=True,
            return_dict=True,
        )
        h = out.last_hidden_state[:, -1].detach()
        past = out.past_key_values
        hs = []
        for t in range(codes.shape[0]):
            hs.append(h.detach())
            if t + 1 >= codes.shape[0]:
                break
            cfg = engine.config
            row = torch.full((1, 1, cfg.n_vq + 1), cfg.audio_pad_token_id, dtype=torch.long, device=engine.device)
            row[:, :, 0] = cfg.speech_generation_start_token_id
            row[:, 0, 1:] = codes[t]
            embeds = model._build_inputs_embeds(row, speaker_emb=spk)
            out = model.semantic_backbone(
                inputs_embeds=embeds.to(dtype=semantic_dtype),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            h = out.last_hidden_state[:, 0].detach()
            past = out.past_key_values
    return hs


def teacher_forced_logits(engine: Any, h: torch.Tensor, target: torch.Tensor):
    """One differentiable full local sequence using true codebook history."""
    model = engine.model
    cfg = engine.config
    local_dtype = next(model.acoustic_decoder.parameters()).dtype
    cond = h[0].to(dtype=local_dtype)
    sgs = torch.tensor([cfg.speech_generation_start_token_id], device=engine.device)
    start = model.text_embeddings(sgs)[0].to(dtype=local_dtype)
    tokens = [cond, start]
    for k in range(cfg.n_vq - 1):
        true_code = target[k].view(1)
        tokens.append(model.audio_embeddings[k](true_code)[0].to(dtype=local_dtype))
    local_input = torch.stack(tokens).unsqueeze(0)
    local_out = model.acoustic_decoder(local_input)
    # Slot 1 predicts codebook 0; slot k+1 predicts codebook k.
    logits = []
    for k in range(cfg.n_vq):
        head = model.audio_lm_heads[k]
        hidden = local_out[:, k + 1]
        head_dtype = next(head.parameters()).dtype
        logits.append(head(hidden.to(dtype=head_dtype)).float())
    return logits


def compute_loss(engine: Any, hs: list[torch.Tensor], codes: torch.Tensor, collect_stats: bool = False):
    losses = []
    by_frame = []
    by_codebook = [[] for _ in range(engine.config.n_vq)]
    for t, h in enumerate(hs):
        logits = teacher_forced_logits(engine, h, codes[t])
        frame_parts = []
        for k, logits_k in enumerate(logits):
            part = F.cross_entropy(logits_k, codes[t, k].view(1))
            frame_parts.append(part)
            by_codebook[k].append(float(part.detach().cpu()))
        frame_loss = torch.stack(frame_parts).mean()
        losses.append(frame_loss)
        by_frame.append(float(frame_loss.detach().cpu()))
    total = torch.stack(losses).mean()
    if collect_stats:
        return total, by_frame, [float(np.mean(x)) for x in by_codebook]
    return total


def trainable_summary(model: torch.nn.Module) -> tuple[int, int, list[str]]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    return total, trainable, names


def main() -> None:
    parser = argparse.ArgumentParser(description="TEST 4A: overfit one v3 Turbo sample with L_audio")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps phải >= 1")

    path, row = load_row()
    engine = load_engine()
    model = engine.model
    model.eval()

    # Freeze everything, then open only acoustic_decoder.layers.0.
    for param in model.parameters():
        param.requires_grad = False
    target_layer = model.acoustic_decoder.layers[0]
    for param in target_layer.parameters():
        param.requires_grad = True

    # Keep the trainable acoustic layer and tied frozen acoustic heads in FP32
    # for a stable optimizer experiment while semantic h_t remains BF16.
    model.acoustic_decoder.float()
    model.audio_embeddings.float()
    model.audio_lm_heads.float()
    if model.xvec_proj is not None:
        model.xvec_proj.float()
    target_layer.train()

    # Reference follows v3 Turbo enrollment semantics (trimmed to max 8s),
    # while the supervised target is the complete utterance. This keeps the
    # conditioning clip and target sequence conceptually separate.
    speaker_emb, reference_codes_np = engine.prepare_reference(
        str(path), denoise=False, use_ref_codes=True
    )
    target_wav, target_sr = engine._load_mono(str(path), None)
    codes_np = engine._encode_ref_wav(target_wav, target_sr)
    codes = torch.as_tensor(np.asarray(codes_np), dtype=torch.long, device=engine.device)
    text = (row.get("transcript") or "").strip()
    if not text:
        raise RuntimeError("Sample không có transcript.")
    prompt = build_prompt(engine, text, np.asarray(reference_codes_np))
    hs = precompute_h(engine, prompt, codes, speaker_emb)

    total_params, trainable_params, trainable_names = trainable_summary(model)
    print("## Sample")
    print(f"filename: {path.name}")
    print(f"duration_sec: {row.get('duration_sec', '')}")
    print(f"speakerID: {row.get('speakerID', '')}")
    print(f"target_codes: {tuple(codes.shape)}")
    print(f"reference_codes: {tuple(np.asarray(reference_codes_np).shape)}")
    print(f"prompt: {tuple(prompt.shape)}")
    print("\n## Trainable modules")
    print("opened: acoustic_decoder.layers.0")
    print(f"total model params: {total_params:,}")
    print(f"trainable params: {trainable_params:,}")
    print(f"trainable percent: {100.0 * trainable_params / total_params:.4f}%")
    for name in trainable_names:
        print(f"  {name}")
    print(f"\noptimizer: AdamW; lr={args.lr}; weight_decay=0; max_grad_norm={args.max_grad_norm}")
    print("loss: L_audio only; EOS/waveform/speaker/auxiliary loss disabled")
    print("semantic h_t: precomputed with TRUE previous MOSS frames")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.0,
    )
    tracked = next(target_layer.parameters())
    weight_before = tracked.detach().float().clone()
    baseline, baseline_frames, baseline_codebooks = compute_loss(engine, hs, codes, collect_stats=True)
    print("\n## Training curve")
    print(f"step 0: loss={float(baseline.detach().cpu()):.6f}")
    print("baseline mean CE by codebook:")
    for k, value in enumerate(baseline_codebooks):
        print(f"  codebook {k}: {value:.6f}")

    checkpoints = set(CHECKPOINTS)
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(engine, hs, codes)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"NaN/Inf loss at step {step}: {loss}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            args.max_grad_norm,
        )
        if not torch.isfinite(torch.as_tensor(grad_norm)).item():
            raise RuntimeError(f"NaN/Inf gradient at step {step}: {grad_norm}")
        optimizer.step()

        if step in checkpoints or step == args.steps:
            with torch.no_grad():
                checked, frame_losses, codebook_losses = compute_loss(engine, hs, codes, collect_stats=True)
            elapsed = time.perf_counter() - start
            vram = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
            print(
                f"step {step}: loss={float(checked.cpu()):.6f}; "
                f"grad_norm={float(torch.as_tensor(grad_norm)):.6f}; lr={args.lr:.2e}; "
                f"VRAM={vram:.3f}GiB; elapsed={elapsed:.1f}s"
            )

    final_loss, final_frames, final_codebooks = compute_loss(engine, hs, codes, collect_stats=True)
    weight_after = tracked.detach().float().clone()
    delta = float((weight_after - weight_before).abs().max().cpu())
    delta_norm = float((weight_after - weight_before).norm().cpu())
    decrease = float(baseline.detach().cpu() - final_loss.detach().cpu())
    percent = 100.0 * decrease / float(baseline.detach().cpu())
    print("\n## Final")
    print(f"baseline_loss: {float(baseline.detach().cpu()):.6f}")
    print(f"final_loss: {float(final_loss.detach().cpu()):.6f}")
    print(f"absolute_decrease: {decrease:.6f}")
    print(f"percent_decrease: {percent:.2f}%")
    print(f"tracked_parameter_max_abs_delta: {delta:.9f}")
    print(f"tracked_parameter_delta_norm: {delta_norm:.9f}")
    print("final mean CE by codebook:")
    for k, value in enumerate(final_codebooks):
        print(f"  codebook {k}: {value:.6f}")
    print("\nNo checkpoint was saved and no upstream source was changed.")


if __name__ == "__main__":
    main()
