"""TEST 5: small speaker-disjoint HaiPhong LoRA pilot.

This keeps the validated v3 Turbo objective and LoRA targets from TEST 4B-B.
The current preferred CSV has one 6-15s clip per speaker, so this pilot uses
6 train speakers and 2 unseen validation speakers with same-utterance reference
fallback, reported explicitly at runtime.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from lora_one_sample_test import LoRALinear
from overfit_one_sample_test import (
    build_prompt,
    compute_loss,
    load_engine,
    precompute_h,
)
from lora_one_sample_ffn_test import inject_ffn_lora


TRAIN_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = TRAIN_DIR.parent
CANDIDATES_DIR = PROJECT_DIR / "vimd_hp_candidates"
PREFERRED_CSV = CANDIDATES_DIR / "metadata_preferred_6_15s.csv"
OUTPUT_DIR = TRAIN_DIR / "output" / "test5_multispeaker_lora"
CHECKPOINTS = (0, 25, 50, 100, 150, 200, 300)


def load_rows() -> list[dict[str, str]]:
    with PREFERRED_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if (CANDIDATES_DIR / r.get("downloaded_file", "")).exists()]
    decisions = {(r.get("decision") or "").strip().casefold() for r in rows}
    if any(decisions - {""}):
        rows = [r for r in rows if (r.get("decision") or "").strip().casefold() == "giữ"]
    if not rows:
        raise RuntimeError("Không có sample hợp lệ sau decision filter.")
    # Keep candidate_042 in train when available for continuity with TEST 4.
    rows.sort(key=lambda r: (0 if r.get("downloaded_file") == "candidate_042.wav" else 1, r.get("downloaded_file", "")))
    return rows


def split_by_speaker(rows: list[dict[str, str]]):
    unique = {}
    for row in rows:
        unique.setdefault(row["speakerID"], row)
    speakers = list(unique)
    if len(speakers) < 8:
        raise RuntimeError(f"Cần ít nhất 8 speaker disjoint, chỉ có {len(speakers)}.")
    train_speakers = speakers[:6]
    val_speakers = speakers[6:8]
    assert not set(train_speakers) & set(val_speakers)
    train = [unique[s] for s in train_speakers]
    val = [unique[s] for s in val_speakers]
    return train, val


def parameter_summary(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, [n for n, p in model.named_parameters() if p.requires_grad]


def prepare_sample(engine, row):
    path = CANDIDATES_DIR / row["downloaded_file"]
    text = (row.get("transcript") or "").strip()
    if not text:
        raise RuntimeError(f"Transcript rỗng: {path.name}")
    speaker_emb, ref_codes_np = engine.prepare_reference(str(path), denoise=False, use_ref_codes=True)
    target_wav, target_sr = engine._load_mono(str(path), None)
    target_codes_np = engine._encode_ref_wav(target_wav, target_sr)
    codes = torch.as_tensor(target_codes_np, dtype=torch.long, device=engine.device)
    prompt = build_prompt(engine, text, ref_codes_np)
    hs = precompute_h(engine, prompt, codes, speaker_emb)
    return {
        "row": row,
        "path": path,
        "reference_codes": np.asarray(ref_codes_np),
        "codes": codes,
        "hs": hs,
        "prompt": prompt,
    }


def evaluate(engine, samples, train_mode: bool):
    values = []
    codebook_values = []
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for sample in samples:
            result = compute_loss(engine, sample["hs"], sample["codes"], collect_stats=True)
            loss, _, codebooks = result
            values.append(loss)
            codebook_values.append(codebooks)
    mean_loss = torch.stack(values).mean()
    mean_cb = np.asarray(codebook_values, dtype=np.float64).mean(axis=0)
    return mean_loss, mean_cb


def lora_state(model):
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if ".lora_A" in name or ".lora_B" in name
    }


def snapshot_lora(model):
    return {name: value.detach().float().clone() for name, value in lora_state(model).items()}


def save_best(model, step: int, value: float, train_value: float, train_speakers, val_speakers):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state(model), OUTPUT_DIR / "adapter_model.pt")
    (OUTPUT_DIR / "adapter_config.json").write_text(json.dumps({
        "rank": 8, "alpha": 16, "dropout": 0.0,
        "targets": [
            "acoustic_decoder.layers.0.attn.qkv",
            "acoustic_decoder.layers.0.attn.o_proj",
            "acoustic_decoder.layers.0.ff_up",
            "acoustic_decoder.layers.0.ff_gate",
            "acoustic_decoder.layers.0.ff_down",
        ],
        "best_step": step,
        "best_validation_loss": value,
        "train_loss_at_best": train_value,
        "train_speakers": train_speakers,
        "validation_speakers": val_speakers,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    engine = load_engine()
    model = engine.model
    model.eval()
    rows = load_rows()
    train_rows, val_rows = split_by_speaker(rows)
    for param in model.parameters():
        param.requires_grad = False
    model.acoustic_decoder.float()
    model.audio_embeddings.float()
    model.audio_lm_heads.float()
    if model.xvec_proj is not None:
        model.xvec_proj.float()
    targets = inject_ffn_lora(model.acoustic_decoder.layers[0], rank=8, alpha=16.0, dropout=0.0)
    for name, param in model.named_parameters():
        param.requires_grad = ".lora_A" in name or ".lora_B" in name

    print("## Dataset split")
    print(f"preferred rows available: {len(rows)}")
    print(f"train speakers: {[r['speakerID'] for r in train_rows]}")
    print(f"validation speakers: {[r['speakerID'] for r in val_rows]}")
    print(f"train samples: {len(train_rows)}; validation samples: {len(val_rows)}")
    print(f"speaker overlap: {len(set(r['speakerID'] for r in train_rows) & set(r['speakerID'] for r in val_rows))}")
    print("reference policy: same utterance as target (preferred CSV has one clip per speaker); no cross-utterance clips available")

    train_samples = [prepare_sample(engine, row) for row in train_rows]
    val_samples = [prepare_sample(engine, row) for row in val_rows]
    total, trainable, names = parameter_summary(model)
    print("\n## Sample distribution")
    for split, samples in (("train", train_samples), ("validation", val_samples)):
        for sample in samples:
            row = sample["row"]
            print(f"{split} | {row['speakerID']} | {row['downloaded_file']} | {row.get('duration_sec','')}s | target_frames={sample['codes'].shape[0]}")
    print("\n## LoRA")
    print(f"targets: {targets}")
    print(f"total params: {total:,}; trainable: {trainable:,}; trainable percent: {100.0*trainable/total:.6f}%")
    print("config: rank=8 alpha=16 dropout=0 lr=1e-4 max_grad_norm=1.0")
    print("loss: L_audio only; EOS/waveform/speaker/auxiliary losses disabled")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0)
    lora_before = snapshot_lora(model)
    base_before = {
        "acoustic_qkv": model.acoustic_decoder.layers[0].attn.qkv.base.weight.detach().float().clone(),
        "acoustic_ff_up": model.acoustic_decoder.layers[0].ff_up.base.weight.detach().float().clone(),
        "semantic": model.semantic_backbone.layers[0].self_attn.q_proj.weight.detach().float().clone(),
        "audio_head": model.audio_lm_heads[0].weight.detach().float().clone(),
        "xvec": model.xvec_proj[0].weight.detach().float().clone() if model.xvec_proj is not None else None,
    }
    train_base, train_cb = evaluate(engine, train_samples, False)
    val_base, val_cb = evaluate(engine, val_samples, False)
    print("\n## Baseline validation")
    print(f"train baseline L_audio: {float(train_base.cpu()):.6f}")
    print(f"validation baseline L_audio: {float(val_base.cpu()):.6f}")
    for sample in val_samples:
        with torch.no_grad():
            one_val, _, _ = compute_loss(engine, sample["hs"], sample["codes"], collect_stats=True)
        print(f"validation speaker {sample['row']['speakerID']}: baseline={float(one_val.cpu()):.6f}")
    print("\n## Training curve")
    print(f"step 0: train={float(train_base.cpu()):.6f}; val={float(val_base.cpu()):.6f}")
    print("step 0 train codebooks:", ", ".join(f"{v:.6f}" for v in train_cb))

    best_val = float(val_base.cpu())
    best_step = 0
    best_train = float(train_base.cpu())
    start = time.perf_counter()
    for step in range(1, 301):
        sample = train_samples[(step - 1) % len(train_samples)]
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(engine, sample["hs"], sample["codes"])
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"NaN/Inf train loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        if not torch.isfinite(torch.as_tensor(grad_norm)).item():
            raise RuntimeError(f"NaN/Inf gradient at step {step}")
        optimizer.step()
        if step in CHECKPOINTS[1:] or step == 300:
            train_value, train_cb_now = evaluate(engine, train_samples, False)
            val_value, val_cb_now = evaluate(engine, val_samples, False)
            train_float, val_float = float(train_value.cpu()), float(val_value.cpu())
            vram = torch.cuda.memory_allocated()/(1024**3) if torch.cuda.is_available() else 0.0
            print(f"step {step}: train={train_float:.6f}; val={val_float:.6f}; grad_norm={float(torch.as_tensor(grad_norm)):.6f}; lr=1.00e-04; VRAM={vram:.3f}GiB; elapsed={time.perf_counter()-start:.1f}s")
            if val_float < best_val:
                best_val, best_step, best_train = val_float, step, train_float
                save_best(model, step, val_float, train_float, [r["speakerID"] for r in train_rows], [r["speakerID"] for r in val_rows])
            if step in (100, 300):
                print(f"step {step} train codebooks: " + ", ".join(f"{v:.6f}" for v in train_cb_now))

    base_after = {
        "acoustic_qkv": model.acoustic_decoder.layers[0].attn.qkv.base.weight.detach().float(),
        "acoustic_ff_up": model.acoustic_decoder.layers[0].ff_up.base.weight.detach().float(),
        "semantic": model.semantic_backbone.layers[0].self_attn.q_proj.weight.detach().float(),
        "audio_head": model.audio_lm_heads[0].weight.detach().float(),
        "xvec": model.xvec_proj[0].weight.detach().float() if model.xvec_proj is not None else None,
    }
    final_lora = snapshot_lora(model)
    print("\n## Parameter verification")
    for name in base_before:
        if base_before[name] is not None:
            print(f"base {name} max_abs_delta: {float((base_after[name]-base_before[name]).abs().max()):.12f}")
    for name, before in lora_before.items():
        delta = float((final_lora[name] - before).abs().max())
        print(f"LoRA {name} max_abs_delta: {delta:.12f}")
    print("validation final by speaker:")
    for sample in val_samples:
        with torch.no_grad():
            one_val, _, _ = compute_loss(engine, sample["hs"], sample["codes"], collect_stats=True)
        print(f"  {sample['row']['speakerID']}: {float(one_val.cpu()):.6f}")
    print("\n## Final")
    print(f"best_step: {best_step}")
    print(f"best_validation_loss: {best_val:.6f}")
    print(f"train_loss_at_best: {best_train:.6f}")
    print(f"validation_absolute_reduction: {float(val_base.cpu())-best_val:.6f}")
    print(f"validation_percent_reduction: {100.0*(float(val_base.cpu())-best_val)/float(val_base.cpu()):.2f}%")
    print(f"best adapter output: {OUTPUT_DIR if OUTPUT_DIR.exists() else 'not saved'}")
    print("No TEST 6/inference accent evaluation was started.")


if __name__ == "__main__":
    main()
