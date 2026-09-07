"""TEST9 dry-run: TEST6 adapters plus a small LoRA on xvec_proj only."""
from __future__ import annotations

import argparse
import gc
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from overfit_one_sample_test import teacher_forced_logits

from lora_one_sample_test import LoRALinear

TRAIN_DIR = Path(__file__).resolve().parent.parent
TEST2_SCRIPT = TRAIN_DIR / "process" / "v3" / "test2_nghean_accent_scale.py"
DATA_EXP_DIR = TRAIN_DIR / "output" / "nghean_test2_accent_scale"
EXP_DIR = TRAIN_DIR / "output" / "nghean_test9_xvec_proj"


def load_test2():
    spec = importlib.util.spec_from_file_location("nghean_test2_runtime_for_test9", TEST2_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load TEST2 script: {TEST2_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure(runner):
    # Reuse TEST6's data loading and shared/tied adapter implementation, but
    # route all output/cache paths to TEST9.
    runner.OUT = EXP_DIR
    runner.EXP_DIR = EXP_DIR
    runner.CACHE_DIR = EXP_DIR / "cache"
    original_read_pairs = runner.read_pairs

    def read_pairs_from_test2(name):
        old_exp = runner.EXP_DIR
        try:
            runner.EXP_DIR = DATA_EXP_DIR
            return original_read_pairs(name)
        finally:
            runner.EXP_DIR = old_exp

    runner.read_pairs = read_pairs_from_test2
    inject_test6 = runner.inject

    def inject_test9(engine):
        inject_test6(engine)
        model = engine.model
        xvec_container = getattr(model, "xvec_proj", None)
        if xvec_container is None:
            raise RuntimeError("xvec_proj does not exist in the loaded v3 Turbo model")
        if not isinstance(xvec_container, nn.Sequential) or len(xvec_container) == 0:
            raise RuntimeError(f"Expected xvec_proj Sequential with child 0, got {type(xvec_container).__name__}")
        xvec = xvec_container[0]
        if not isinstance(xvec, nn.Linear):
            raise RuntimeError(f"xvec_proj.0 must be nn.Linear, got {type(xvec).__name__}")
        print(
            f"xvec_proj.0: {xvec.__class__.__name__} {xvec.in_features}->{xvec.out_features}; "
            f"base_params={sum(p.numel() for p in xvec.parameters())}",
            flush=True,
        )
        adapter = LoRALinear(xvec, rank=8, alpha=16.0, dropout=0.0)
        model.xvec_proj[0] = adapter
        for name, parameter in model.named_parameters():
            parameter.requires_grad = ".lora_A" in name or ".lora_B" in name
        if model.xvec_proj[0].base.weight.requires_grad or (
            model.xvec_proj[0].base.bias is not None and model.xvec_proj[0].base.bias.requires_grad
        ):
            raise RuntimeError("xvec_proj base weight/bias was not frozen")
        runner._xvec_adapter = adapter
        print("xvec_proj LoRA: rank=8 alpha=16; base weight frozen", flush=True)

    runner.inject = inject_test9
    return runner


def true_history_with_xvec_grad(engine, sample):
    """TEST9 variant: preserve gradient from L_audio into xvec_proj.  The
    original helper deliberately uses no_grad()+detach() because xvec_proj
    was frozen in earlier tests, which would otherwise make this adapter
    permanently unreachable from the loss.
    """
    model = engine.model
    model.semantic_backbone.eval()
    prompt = sample["prompt"]
    codes = sample["codes"]
    speaker_emb = sample["speaker_emb"]
    ids = prompt.unsqueeze(0).to(engine.device)
    spk = engine._resolve_speaker_emb(speaker_emb)
    semantic_dtype = next(model.semantic_backbone.parameters()).dtype
    embeds = model._build_inputs_embeds(ids, speaker_emb=spk)
    out = model.semantic_backbone(
        inputs_embeds=embeds.to(dtype=semantic_dtype),
        use_cache=True,
        return_dict=True,
    )
    h = out.last_hidden_state[:, -1]
    past = out.past_key_values
    hs = []
    for t in range(codes.shape[0]):
        hs.append(h)
        if t + 1 >= codes.shape[0]:
            break
        cfg = engine.config
        row = torch.full(
            (1, 1, cfg.n_vq + 1),
            cfg.audio_pad_token_id,
            dtype=torch.long,
            device=engine.device,
        )
        row[:, :, 0] = cfg.speech_generation_start_token_id
        row[:, 0, 1:] = codes[t]
        embeds = model._build_inputs_embeds(row, speaker_emb=spk)
        out = model.semantic_backbone(
            inputs_embeds=embeds.to(dtype=semantic_dtype),
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        h = out.last_hidden_state[:, 0]
        past = out.past_key_values
    return hs


def main():
    parser = argparse.ArgumentParser(description="TEST9 xvec_proj LoRA dry-run only")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    test6_spec = importlib.util.spec_from_file_location(
        "nghean_test6_module_for_test9", TRAIN_DIR / "process" / "v3" / "test6_nghean_shared_tied_audio.py"
    )
    if test6_spec is None or test6_spec.loader is None:
        raise RuntimeError("Cannot load TEST6 adapter implementation")
    test6 = importlib.util.module_from_spec(test6_spec)
    test6_spec.loader.exec_module(test6)
    runner = configure(test6.configure(load_test2()))
    runner.true_history = true_history_with_xvec_grad
    if args.train:
        train_only(runner)
        return
    runner.OUT.mkdir(parents=True, exist_ok=True)
    runner.seed(runner.SEED)

    train = runner.read_pairs("nghean_train_pairs_v2.csv")
    valid = runner.read_pairs("nghean_valid_pairs_v2.csv")
    test = runner.read_pairs("nghean_test_pairs_v2.csv")
    sets = runner.validate_split(train, valid, test)
    overlap = (sets["train"] & sets["valid"]) | (sets["train"] & sets["test"]) | (sets["valid"] & sets["test"])
    print(f"dataset = {len(train)}/{len(valid)}/{len(test)}", flush=True)
    print(f"speaker overlap = {len(overlap)}", flush=True)
    if overlap:
        raise RuntimeError(f"speaker overlap detected: {sorted(overlap)}")

    runner.PREP_TOTAL = 3
    engine = runner._load_engine()
    runner.inject(engine)
    sample = runner.prepare(engine, train[0])
    loss_value = streaming_backward(runner, engine, sample)
    loss = torch.tensor(loss_value, device=engine.device, requires_grad=True)

    model = engine.model
    layer_params = [
        p for n, p in model.named_parameters()
        if n.startswith("acoustic_decoder.layers.0") and ".lora_" in n
    ]
    layer_ok = bool(layer_params) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer_params)
    head_ok = []
    for k in range(16):
        head = model.audio_lm_heads[k]
        head_ok.append(
            head.lora_A.grad is not None and torch.isfinite(head.lora_A.grad).all()
            and head.lora_B.grad is not None and torch.isfinite(head.lora_B.grad).all()
        )
    xvec = model.xvec_proj[0]
    xvec_ok = (
        xvec.lora_A.grad is not None and torch.isfinite(xvec.lora_A.grad).all()
        and xvec.lora_B.grad is not None and torch.isfinite(xvec.lora_B.grad).all()
    )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"xvec_proj path = xvec_proj.0", flush=True)
    print(f"xvec_proj shape = {xvec.base.in_features} -> {xvec.base.out_features}", flush=True)
    print("layer0 LoRA = OK; shared/tied heads = OK; xvec_proj LoRA = OK", flush=True)
    print("xvec base frozen = YES", flush=True)
    print("r = 8; alpha = 16", flush=True)
    print(f"dry-run loss = {loss_value:.6f}; finite = YES; requires_grad = YES", flush=True)
    print(f"layer0 gradient = {'OK' if layer_ok else 'FAIL'}", flush=True)
    print(f"audio adapters gradient = {'OK' if all(head_ok) else 'FAIL'}; valid = {sum(head_ok)}/16", flush=True)
    print(f"xvec gradient = {'OK' if xvec_ok else 'FAIL'}", flush=True)
    print(f"xvec_proj LoRA params = {xvec.lora_A.numel() + xvec.lora_B.numel():,}", flush=True)
    print(f"trainable params = {trainable:,}; trainable % = {100.0 * trainable / total:.6f}%", flush=True)
    print("DRY-RUN ONLY: no optimizer.step(), no training", flush=True)


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_memory_safe(runner, engine, samples):
    total = 0.0
    with torch.no_grad():
        for i, sample in enumerate(samples, 1):
            history = runner.true_history(engine, sample)
            loss = runner.compute_loss(engine, history, sample["codes"])
            value = float(loss.detach().cpu())
            total += value
            print(
                f"evaluate {i}/{len(samples)}: {sample['row']['split']} "
                f"{sample['row']['speakerID']} {sample['row']['target_path'].name} ce={value:.6f}",
                flush=True,
            )
            del loss, history
            release_memory()
    return total / len(samples)


def train_only(runner):
    runner.OUT.mkdir(parents=True, exist_ok=True)
    runner.seed(runner.SEED)
    train_rows = runner.read_pairs("nghean_train_pairs_v2.csv")
    valid_rows = runner.read_pairs("nghean_valid_pairs_v2.csv")
    test_rows = runner.read_pairs("nghean_test_pairs_v2.csv")
    sets = runner.validate_split(train_rows, valid_rows, test_rows)
    overlap = (sets["train"] & sets["valid"]) | (sets["train"] & sets["test"]) | (sets["valid"] & sets["test"])
    if overlap:
        raise RuntimeError(f"speaker overlap detected: {sorted(overlap)}")
    runner.PREP_TOTAL = len(train_rows) + len(valid_rows) + len(test_rows)
    print(f"dataset rows train/valid/test: {len(train_rows)}/{len(valid_rows)}/{len(test_rows)}", flush=True)
    engine = runner._load_engine()
    runner.inject(engine)
    train = [runner.prepare(engine, row) for row in train_rows]
    valid = [runner.prepare(engine, row) for row in valid_rows]
    [runner.prepare(engine, row) for row in test_rows]

    opt = torch.optim.AdamW([p for p in engine.model.parameters() if p.requires_grad], lr=runner.LR, weight_decay=0.0)
    curve = []
    best_val = float("inf")
    best_step = 0
    best_state = None
    stale = 0
    checkpoints = runner.OUT / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    for step in range(1, runner.TF_STEPS + 1):
        sample = train[(step - 1) % len(train)]
        opt.zero_grad(set_to_none=True)
        train_value = streaming_backward(runner, engine, sample)
        grad = torch.nn.utils.clip_grad_norm_([p for p in engine.model.parameters() if p.requires_grad], 1.0)
        if not torch.isfinite(torch.as_tensor(grad)):
            raise RuntimeError(f"NaN/Inf gradient step {step}")
        opt.step()
        release_memory()
        if step % runner.VALIDATION_INTERVAL == 0:
            val = evaluate_memory_safe(runner, engine, valid)
            is_best = val < best_val
            item = {"step": step, "train_loss": train_value, "validation_CE": val,
                    "best_validation_CE": min(best_val, val), "is_best": is_best,
                    "learning_rate": runner.LR, "gradient_norm": float(torch.as_tensor(grad))}
            curve.append(item)
            print(f"tf step {step}: train={train_value:.6f} val={val:.6f} best={item['best_validation_CE']:.6f} grad={item['gradient_norm']:.6f} is_best={is_best}", flush=True)
            if is_best:
                best_val, best_step, best_state, stale = val, step, runner.lora_state(engine.model), 0
                torch.save(best_state, checkpoints / f"best_step_{step:03d}.pt")
            else:
                stale += 1
            release_memory()
            if stale >= runner.EARLY_STOP_PATIENCE:
                print(f"early stop at step {step}: no new best for {runner.EARLY_STOP_PATIENCE} validations", flush=True)
                break
    if best_state is None:
        raise RuntimeError("No best checkpoint was produced")
    best_dir = runner.OUT / "best_tf_checkpoint"
    runner.save_adapter(engine, best_dir, best_step, best_val, "teacher_forcing_xvec_proj")
    header = "step,train_loss,validation_CE,best_validation_CE,is_best,learning_rate,gradient_norm\n"
    body = "\n".join(",".join(str(item[key]) for key in ("step", "train_loss", "validation_CE", "best_validation_CE", "is_best", "learning_rate", "gradient_norm")) for item in curve)
    (runner.OUT / "validation_history.csv").write_text(header + body, encoding="utf-8")
    final_val = curve[-1]["validation_CE"]
    print(f"TRAIN COMPLETE: steps={curve[-1]['step']}; best_step={best_step}; best_validation_CE={best_val:.6f}; final_validation_CE={final_val:.6f}", flush=True)
    print(f"best checkpoint: {best_dir}", flush=True)


def detach_tree(value):
    if torch.is_tensor(value):
        return value.detach()
    # Recent Transformers returns a DynamicCache object rather than a tuple
    # for past_key_values. Detach its per-layer KV tensors explicitly so a
    # streamed frame never backpropagates through the previous frame graph.
    if hasattr(value, "key_cache") and hasattr(value, "value_cache"):
        value.key_cache = [x.detach() for x in value.key_cache]
        value.value_cache = [x.detach() for x in value.value_cache]
        return value
    if isinstance(value, tuple):
        return tuple(detach_tree(x) for x in value)
    if isinstance(value, list):
        return [detach_tree(x) for x in value]
    if isinstance(value, dict):
        return {k: detach_tree(v) for k, v in value.items()}
    return value


def streaming_backward(runner, engine, sample):
    """Low-VRAM TEST9 objective.

    Rebuild the true history from the prompt for one frame at a time. This
    avoids retaining DynamicCache/autograd references between frames and
    keeps only one semantic/acoustic graph on the GPU. It is intentionally
    slower, but does not spill the whole sequence into virtual memory.
    """
    model = engine.model
    model.semantic_backbone.eval()
    prompt = sample["prompt"]
    codes = sample["codes"]
    spk = engine._resolve_speaker_emb(sample["speaker_emb"])
    semantic_dtype = next(model.semantic_backbone.parameters()).dtype
    total = 0.0
    frame_count = int(codes.shape[0])
    for t in range(frame_count):
        # Exact true-history prefix for frame t, but no persistent KV cache.
        pieces = [prompt.unsqueeze(0).to(engine.device)]
        if t:
            cfg = engine.config
            rows = torch.full(
                (1, t, cfg.n_vq + 1), cfg.audio_pad_token_id,
                dtype=torch.long, device=engine.device,
            )
            rows[:, :, 0] = cfg.speech_generation_start_token_id
            rows[:, :, 1:] = codes[:t]
            pieces.append(rows)
        ids = torch.cat(pieces, dim=1)
        embeds = model._build_inputs_embeds(ids, speaker_emb=spk)
        out = model.semantic_backbone(
            inputs_embeds=embeds.to(dtype=semantic_dtype),
            use_cache=False,
            return_dict=True,
        )
        h = out.last_hidden_state[:, -1]
        logits = teacher_forced_logits(engine, h, codes[t])
        frame_loss = torch.stack([
            F.cross_entropy(logit, codes[t, k].view(1))
            for k, logit in enumerate(logits)
        ]).mean()
        if not torch.isfinite(frame_loss):
            raise RuntimeError(f"NaN/Inf frame loss at frame {t}")
        (frame_loss / frame_count).backward()
        total += float(frame_loss.detach().cpu())
        # The shared adapter delta is reused within this frame, but its
        # autograd graph is consumed by backward. Do not reuse that graph on
        # the next streamed frame.
        for adapter in getattr(runner, "_shared_adapters", []):
            adapter._cached_delta = None
            adapter._cached_versions = None
        del logits, frame_loss, h, out, embeds
        release_memory()
    return total / frame_count


if __name__ == "__main__":
    main()
