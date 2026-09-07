"""TEST 4B-A: tiny custom LoRA on the v3 Turbo acoustic attention.

Targets only:
  acoustic_decoder.layers.0.attn.qkv
  acoustic_decoder.layers.0.attn.o_proj

The implementation is local to this test file. It does not use the legacy
0.3B/NeuCodec/PEFT pipeline and does not modify upstream VieNeu source.
"""
from __future__ import annotations

import argparse
import time
from typing import Any

import torch
import torch.nn as nn

from overfit_one_sample_test import (
    CHECKPOINT,
    MOSS_REPO,
    build_prompt,
    compute_loss,
    load_engine,
    load_row,
    precompute_h,
)


class LoRALinear(nn.Module):
    """Frozen Linear plus trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRA target must be nn.Linear, got {type(base).__name__}")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        update = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + update * self.scaling


def inject_lora(layer: nn.Module, rank: int, alpha: float, dropout: float) -> list[str]:
    targets = ["attn.qkv", "attn.o_proj"]
    replaced = []
    for path in targets:
        parent_path, child_name = path.rsplit(".", 1)
        parent = layer
        for part in parent_path.split("."):
            parent = getattr(parent, part)
        base = getattr(parent, child_name)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Expected nn.Linear at {path}, got {type(base).__name__}")
        setattr(parent, child_name, LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(f"acoustic_decoder.layers.0.{path}")
    return replaced


def parameter_summary(model: nn.Module) -> tuple[int, int, list[str]]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    return total, trainable, names


def snapshot_base(model: nn.Module) -> dict[str, torch.Tensor]:
    paths = {
        "acoustic_qkv": model.acoustic_decoder.layers[0].attn.qkv.base.weight,
        "semantic": model.semantic_backbone.layers[0].self_attn.q_proj.weight,
        "audio_head_0": model.audio_lm_heads[0].weight,
    }
    return {name: tensor.detach().float().clone() for name, tensor in paths.items()}


def snapshot_lora(model: nn.Module) -> dict[str, torch.Tensor]:
    layer = model.acoustic_decoder.layers[0]
    return {
        "qkv_A": layer.attn.qkv.lora_A.detach().float().clone(),
        "qkv_B": layer.attn.qkv.lora_B.detach().float().clone(),
        "o_proj_A": layer.attn.o_proj.lora_A.detach().float().clone(),
        "o_proj_B": layer.attn.o_proj.lora_B.detach().float().clone(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TEST 4B-A: one-sample acoustic attention LoRA")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args()
    if args.steps < 1 or args.rank < 1:
        raise SystemExit("--steps và --rank phải >= 1")

    path, row = load_row()
    engine = load_engine()
    model = engine.model
    model.eval()

    # Freeze every base parameter before inserting adapters.
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.acoustic_decoder.float()
    model.audio_embeddings.float()
    model.audio_lm_heads.float()
    if model.xvec_proj is not None:
        model.xvec_proj.float()

    layer = model.acoustic_decoder.layers[0]
    targets = inject_lora(layer, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    # Explicitly enforce the intended trainable set by parameter name.
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (".lora_A" in name or ".lora_B" in name)
    for module in (layer.attn.qkv, layer.attn.o_proj):
        module.train()

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
    print("\n## LoRA")
    print(f"target modules: {targets}")
    print(f"rank: {args.rank}; alpha: {args.alpha}; scaling: {args.alpha / args.rank:.4f}; dropout: {args.dropout}")
    print(f"total model params: {total:,}")
    print(f"trainable LoRA params: {trainable:,}")
    print(f"trainable percent: {100.0 * trainable / total:.6f}%")
    print("trainable parameter names:")
    for name in names:
        print(f"  {name}")
    print("base weights: frozen; EOS/waveform/speaker losses: disabled")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
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

    checkpoints = {0, 1, 5, 10, 20, 50, 100, 150, 200}
    start_time = time.perf_counter()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(engine, hs, codes)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"NaN/Inf loss at step {step}: {loss}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            args.max_grad_norm,
        )
        if not torch.isfinite(torch.as_tensor(grad_norm)).item():
            raise RuntimeError(f"NaN/Inf gradient at step {step}: {grad_norm}")
        optimizer.step()

        if step in checkpoints or step == args.steps:
            with torch.no_grad():
                checked, _, cb = compute_loss(engine, hs, codes, collect_stats=True)
            vram = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
            print(
                f"step {step}: loss={float(checked.cpu()):.6f}; "
                f"grad_norm={float(torch.as_tensor(grad_norm)):.6f}; lr={args.lr:.2e}; "
                f"VRAM={vram:.3f}GiB; elapsed={time.perf_counter() - start_time:.1f}s"
            )

    final_loss, _, final_cb = compute_loss(engine, hs, codes, collect_stats=True)
    base_after = snapshot_base(model)
    lora_after = snapshot_lora(model)
    print("\n## Parameter verification")
    for name in base_before:
        delta = float((base_after[name] - base_before[name]).abs().max())
        print(f"base {name} max_abs_delta: {delta:.12f}")
    for name in lora_before:
        delta = float((lora_after[name] - lora_before[name]).abs().max())
        print(f"LoRA {name} max_abs_delta: {delta:.12f}")

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
    print("\nNo checkpoint was saved and no upstream source was changed.")


if __name__ == "__main__":
    main()
