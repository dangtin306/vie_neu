"""TEST 4B-B: custom rank-8 LoRA on acoustic attention + FFN layer 0.

Targets exactly five runtime nn.Linear modules in the local v3 Turbo model:
attn.qkv, attn.o_proj, ff_up, ff_gate, and ff_down.
"""
from __future__ import annotations

import time

import torch

from lora_one_sample_test import LoRALinear, inject_lora
from overfit_one_sample_test import (
    build_prompt,
    compute_loss,
    load_engine,
    load_row,
    precompute_h,
)


def inject_ffn_lora(layer, rank: int, alpha: float, dropout: float) -> list[str]:
    targets = ["attn.qkv", "attn.o_proj", "ff_up", "ff_gate", "ff_down"]
    replaced = []
    for path in targets:
        parent = layer
        parts = path.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child = parts[-1]
        base = getattr(parent, child)
        if not isinstance(base, torch.nn.Linear):
            raise TypeError(f"Expected nn.Linear at acoustic_decoder.layers.0.{path}, got {type(base).__name__}")
        print(f"runtime module acoustic_decoder.layers.0.{path}: {base.in_features}->{base.out_features}, params={sum(p.numel() for p in base.parameters())}")
        setattr(parent, child, LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(f"acoustic_decoder.layers.0.{path}")
    return replaced


def parameter_summary(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    return total, trainable, names


def snapshot_base(model):
    values = {
        "acoustic_qkv": model.acoustic_decoder.layers[0].attn.qkv.base.weight,
        "acoustic_ff_up": model.acoustic_decoder.layers[0].ff_up.base.weight,
        "semantic": model.semantic_backbone.layers[0].self_attn.q_proj.weight,
        "audio_head_0": model.audio_lm_heads[0].weight,
    }
    return {name: value.detach().float().clone() for name, value in values.items()}


def snapshot_lora(model):
    layer = model.acoustic_decoder.layers[0]
    modules = {
        "qkv": layer.attn.qkv,
        "o_proj": layer.attn.o_proj,
        "ff_up": layer.ff_up,
        "ff_gate": layer.ff_gate,
        "ff_down": layer.ff_down,
    }
    result = {}
    for name, module in modules.items():
        result[f"{name}_A"] = module.lora_A.detach().float().clone()
        result[f"{name}_B"] = module.lora_B.detach().float().clone()
    return result


def main() -> None:
    path, row = load_row()
    engine = load_engine()
    model = engine.model
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.acoustic_decoder.float()
    model.audio_embeddings.float()
    model.audio_lm_heads.float()
    if model.xvec_proj is not None:
        model.xvec_proj.float()

    layer = model.acoustic_decoder.layers[0]
    targets = inject_ffn_lora(layer, rank=8, alpha=16.0, dropout=0.0)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (".lora_A" in name or ".lora_B" in name)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (".lora_A" not in name and ".lora_B" not in name):
            raise RuntimeError(f"Unexpected trainable parameter: {name}")
    layer.train()

    speaker_emb, reference_codes_np = engine.prepare_reference(str(path), denoise=False, use_ref_codes=True)
    target_wav, target_sr = engine._load_mono(str(path), None)
    target_codes_np = engine._encode_ref_wav(target_wav, target_sr)
    codes = torch.as_tensor(target_codes_np, dtype=torch.long, device=engine.device)
    text = (row.get("transcript") or "").strip()
    if not text:
        raise RuntimeError("Sample không có transcript.")
    prompt = build_prompt(engine, text, reference_codes_np)
    hs = precompute_h(engine, prompt, codes, speaker_emb)

    total, trainable, names = parameter_summary(model)
    print("## Sample")
    print(f"filename: {path.name}")
    print(f"speakerID: {row.get('speakerID', '')}")
    print(f"duration_sec: {row.get('duration_sec', '')}")
    print(f"reference_codes: {tuple(reference_codes_np.shape)}")
    print(f"target_codes: {tuple(codes.shape)}")
    print("\n## LoRA targets")
    print(f"targets: {targets}")
    print("rank=8; alpha=16; scaling=2; dropout=0; bias=none")
    print(f"total model params: {total:,}")
    print(f"trainable LoRA params: {trainable:,}")
    print(f"trainable percent: {100.0 * trainable / total:.6f}%")
    print("trainable names:")
    for name in names:
        print(f"  {name}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.0,
    )
    base_before = snapshot_base(model)
    lora_before = snapshot_lora(model)
    baseline, _, baseline_cb = compute_loss(engine, hs, codes, collect_stats=True)
    print("\n## Training curve")
    print(f"step 0: loss={float(baseline.detach().cpu()):.6f}")
    print("step 0 mean CE by codebook:")
    for k, value in enumerate(baseline_cb):
        print(f"  codebook {k}: {value:.6f}")

    checkpoints = {1, 5, 10, 20, 50, 100, 150, 200}
    start = time.perf_counter()
    for step in range(1, 201):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(engine, hs, codes)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"NaN/Inf loss at step {step}: {loss}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        if not torch.isfinite(torch.as_tensor(grad_norm)).item():
            raise RuntimeError(f"NaN/Inf gradient at step {step}: {grad_norm}")
        optimizer.step()
        if step in checkpoints:
            with torch.no_grad():
                checked, _, _ = compute_loss(engine, hs, codes, collect_stats=True)
            vram = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
            print(f"step {step}: loss={float(checked.cpu()):.6f}; grad_norm={float(torch.as_tensor(grad_norm)):.6f}; lr=1.00e-04; VRAM={vram:.3f}GiB; elapsed={time.perf_counter()-start:.1f}s")

    final_loss, _, final_cb = compute_loss(engine, hs, codes, collect_stats=True)
    base_after = snapshot_base(model)
    lora_after = snapshot_lora(model)
    print("\n## Parameter verification")
    for name in base_before:
        print(f"base {name} max_abs_delta: {float((base_after[name]-base_before[name]).abs().max()):.12f}")
    for name in lora_before:
        print(f"LoRA {name} max_abs_delta: {float((lora_after[name]-lora_before[name]).abs().max()):.12f}")

    baseline_value = float(baseline.detach().cpu())
    final_value = float(final_loss.detach().cpu())
    decrease = baseline_value - final_value
    print("\n## Final")
    print(f"baseline_loss: {baseline_value:.6f}")
    print(f"final_loss: {final_value:.6f}")
    print(f"absolute_decrease: {decrease:.6f}")
    print(f"percent_decrease: {100.0 * decrease / baseline_value:.2f}%")
    print("final mean CE by codebook:")
    for k, value in enumerate(final_cb):
        print(f"  codebook {k}: {value:.6f}")
    print("No checkpoint saved; no upstream source changed.")


if __name__ == "__main__":
    main()
