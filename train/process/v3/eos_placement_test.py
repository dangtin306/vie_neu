"""Runtime EOS placement test for local VieNeu-TTS v3 Turbo.

Measures speech-start/EOS probabilities at h_0..h_{T-1} and h_T while
teacher-forcing true MOSS frames. It also checks whether text EOS logits at
AcousticDecoder slot 0 depend on unrolled codebooks 0..15.

No training, loss update, backward, optimizer, LoRA, or checkpoint is used.
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
    if list((TEXT_PROCESS_PYTHON / "sea_g2p").glob("sea_g2p_rs*.pyd")):
        sys.path.insert(0, str(TEXT_PROCESS_PYTHON))


def select_rows(limit: int) -> list[dict[str, str]]:
    with PREFERRED_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [
        r for r in rows
        if (CANDIDATES_DIR / r.get("downloaded_file", "")).exists()
    ]
    rows.sort(key=lambda r: (0 if (r.get("decision") or "").strip().casefold() == "giữ" else 1, r.get("downloaded_file", "")))
    selected: list[dict[str, str]] = []
    speakers: set[str] = set()
    for row in rows:
        speaker = row.get("speakerID", "")
        if speaker in speakers:
            continue
        selected.append(row)
        speakers.add(speaker)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        # If the preferred file has fewer unique speakers, fill remaining rows.
        used = {r.get("downloaded_file") for r in selected}
        for row in rows:
            if row.get("downloaded_file") not in used:
                selected.append(row)
                used.add(row.get("downloaded_file"))
                if len(selected) >= limit:
                    break
    if not selected:
        raise RuntimeError("Không có WAV Hải Phòng hợp lệ.")
    return selected


def load_engine():
    from vieneu._v3_turbo_engine.inference_v3_turbo import VieNeuTTSv3Turbo

    return VieNeuTTSv3Turbo(
        checkpoint_path=CHECKPOINT,
        model_subfolder=MODEL_SUBFOLDER,
        moss_tokenizer_path=MOSS_REPO,
        device="auto",
        dtype="auto",
    )


def cache_length(past: Any) -> int:
    if hasattr(past, "get_seq_length"):
        return int(past.get_seq_length())
    return int(past[0][0].shape[2])


def build_prompt(engine: Any, text: str, codes: np.ndarray) -> torch.Tensor:
    from vieneu_utils.phonemize_text import phonemize_text_with_emotions

    phones = phonemize_text_with_emotions(text)
    return engine._build_prompt_2d(phones, None, codes, engine._resolve_style_id())


def prefill(engine: Any, prompt: torch.Tensor, speaker_emb: np.ndarray | None):
    model = engine.model
    ids = prompt.unsqueeze(0).to(engine.device)
    spk = engine._resolve_speaker_emb(speaker_emb)
    embeds = model._build_inputs_embeds(ids, speaker_emb=spk)
    out = model.semantic_backbone(inputs_embeds=embeds, use_cache=True, return_dict=True)
    return out.last_hidden_state[:, -1], out.past_key_values


def append_frame(engine: Any, past: Any, frame: torch.Tensor, speaker_emb: np.ndarray | None):
    cfg = engine.config
    model = engine.model
    row = torch.full((1, 1, cfg.n_vq + 1), cfg.audio_pad_token_id, dtype=torch.long, device=engine.device)
    row[:, :, 0] = cfg.speech_generation_start_token_id
    row[:, 0, 1:] = frame
    spk = engine._resolve_speaker_emb(speaker_emb)
    embeds = model._build_inputs_embeds(row, speaker_emb=spk)
    out = model.semantic_backbone(inputs_embeds=embeds, past_key_values=past, use_cache=True, return_dict=True)
    return out.last_hidden_state[:, 0], out.past_key_values


def text_logits_at_h(engine: Any, h: torch.Tensor, true_frame: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return logits before and after true codebook unrolling at local slot 0."""
    model = engine.model
    cfg = engine.config
    local_dtype = next(model.acoustic_decoder.parameters()).dtype
    sgs = torch.tensor([cfg.speech_generation_start_token_id], device=engine.device)
    cond = h[0].to(dtype=local_dtype)
    start = model.text_embeddings(sgs)[0].to(dtype=local_dtype)
    initial = torch.stack([cond, start]).unsqueeze(0)
    initial_out = model.acoustic_decoder(initial)
    logits_a = model.text_lm_head(initial_out[0, 0]).float()
    if true_frame is None:
        return logits_a, logits_a

    tokens = [cond, start]
    for k in range(cfg.n_vq):
        current = torch.stack(tokens).unsqueeze(0)
        if k < cfg.n_vq - 1:
            true_code = true_frame[k].view(1)
            tokens.append(model.audio_embeddings[k](true_code)[0].to(dtype=local_dtype))
        else:
            # Materialize the same 17-token local sequence used by the full
            # unroll before inspecting slot 0.
            full = current
    full_out = model.acoustic_decoder(full)
    logits_b = model.text_lm_head(full_out[0, 0]).float()
    return logits_a, logits_b


def probs(engine: Any, logits: torch.Tensor) -> tuple[float, float, float, float]:
    cfg = engine.config
    p = F.softmax(logits, dim=-1)
    p5 = float(p[cfg.speech_generation_start_token_id].cpu())
    p6 = float(p[cfg.speech_generation_end_token_id].cpu())
    l5 = float(logits[cfg.speech_generation_start_token_id].cpu())
    l6 = float(logits[cfg.speech_generation_end_token_id].cpu())
    return l5, l6, p5, p6


def ce_for_labels(engine: Any, records: list[tuple[torch.Tensor, torch.Tensor]], eos_at: int) -> tuple[float, float]:
    cfg = engine.config
    a_losses = []
    b_losses = []
    for t, (logits, _) in enumerate(records):
        label_a = cfg.speech_generation_end_token_id if t == eos_at else cfg.speech_generation_start_token_id
        a_losses.append(F.cross_entropy(logits.view(1, -1), torch.tensor([label_a], device=engine.device)))
    # h_T is supplied separately by the caller for hypothesis B.
    return float(torch.stack(a_losses).mean().cpu()), float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="v3 Turbo EOS placement runtime test")
    parser.add_argument("--clips", type=int, default=5, help="number of distinct-speaker clips")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all target frames")
    args = parser.parse_args()
    add_imports()
    rows = select_rows(max(1, args.clips))
    engine = load_engine()
    engine.model.eval()
    cfg = engine.config
    print(f"Checkpoint: {CHECKPOINT}/{MODEL_SUBFOLDER}")
    print(f"EOS id: {cfg.speech_generation_end_token_id}; continue/start id: {cfg.speech_generation_start_token_id}")
    print(f"Clips: {len(rows)}; no_grad runtime measurement; no training/update")

    summary = []
    for clip_index, row in enumerate(rows, 1):
        path = CANDIDATES_DIR / row["downloaded_file"]
        text = (row.get("transcript") or "").strip()
        if not text:
            print(f"SKIP {path.name}: transcript rỗng")
            continue
        with torch.no_grad():
            speaker_emb, codes_np = engine.prepare_reference(str(path), denoise=False, use_ref_codes=True)
            codes = torch.as_tensor(np.asarray(codes_np), dtype=torch.long, device=engine.device)
            prompt = build_prompt(engine, text, np.asarray(codes_np))
            h, past = prefill(engine, prompt, speaker_emb)
            target_frames = int(codes.shape[0])
            n_frames = min(target_frames, args.max_frames) if args.max_frames else target_frames
            records: list[tuple[torch.Tensor, torch.Tensor]] = []
            max_diff = 0.0
            near_final: list[tuple[int, float, float, float, float, float]] = []
            for t in range(n_frames):
                logits_a, logits_b = text_logits_at_h(engine, h, codes[t])
                diff = float((logits_a - logits_b).abs().max().cpu())
                max_diff = max(max_diff, diff)
                l5, l6, p5, p6 = probs(engine, logits_a)
                records.append((logits_a, logits_b))
                if t >= max(0, n_frames - 8):
                    near_final.append((t, l5, l6, p5, p6, l6 - l5))
                if t + 1 < n_frames:
                    h, past = append_frame(engine, past, codes[t], speaker_emb)

            # h_T is obtained only after feeding the final true frame back.
            h_t, past_t = append_frame(engine, past, codes[n_frames - 1], speaker_emb)
            logits_t, logits_t_full = text_logits_at_h(engine, h_t, None)
            l5t, l6t, p5t, p6t = probs(engine, logits_t)

            # Hypothesis A: EOS at h_{T-1}; hypothesis B: EOS at h_T.
            label5 = cfg.speech_generation_start_token_id
            label6 = cfg.speech_generation_end_token_id
            loss_a = torch.stack([
                F.cross_entropy(logits.view(1, -1), torch.tensor([label6 if t == n_frames - 1 else label5], device=engine.device))
                for t, (logits, _) in enumerate(records)
            ]).mean()
            loss_b = torch.stack([
                *[
                    F.cross_entropy(logits.view(1, -1), torch.tensor([label5], device=engine.device))
                    for logits, _ in records
                ],
                F.cross_entropy(logits_t.view(1, -1), torch.tensor([label6], device=engine.device)),
            ]).mean()

        print(f"\n[{clip_index}/{len(rows)}] {path.name} speaker={row.get('speakerID', '')} frames={target_frames} tested={n_frames}")
        print(f"  prompt={tuple(prompt.shape)} cache_final={cache_length(past_t)} max|logits_A-logits_B|={max_diff:.8f}")
        print("  frame | logit_5 | logit_6 | P(5) | P(6) | margin(6-5)")
        for t, l5, l6, p5, p6, margin in near_final:
            print(f"  {t:5d} | {l5:8.4f} | {l6:8.4f} | {p5:6.4f} | {p6:6.4f} | {margin:10.4f}")
        print(f"  h_T  | {l5t:8.4f} | {l6t:8.4f} | {p5t:6.4f} | {p6t:6.4f} | {l6t-l5t:10.4f}")
        print(f"  hypothesis A mean CE: {float(loss_a.cpu()):.6f}")
        print(f"  hypothesis B mean CE: {float(loss_b.cpu()):.6f}")
        summary.append((path.name, row.get("speakerID", ""), n_frames, p6 if near_final else 0.0, p6t, max_diff, float(loss_a.cpu()), float(loss_b.cpu())))

    print("\n## Summary")
    print("clip | speaker | frames | P(EOS@last frame) | P(EOS@h_T) | max local diff | CE-A | CE-B")
    for item in summary:
        name, speaker, n, plast, pt, diff, ca, cb = item
        print(f"{name} | {speaker} | {n} | {plast:.6f} | {pt:.6f} | {diff:.8f} | {ca:.6f} | {cb:.6f}")
    print("\nNo backward(), optimizer.step(), LoRA/PEFT, weight update, or checkpoint save was performed.")


if __name__ == "__main__":
    main()
