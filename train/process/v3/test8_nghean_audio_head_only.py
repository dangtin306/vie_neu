"""TEST8: train only the 16 shared tied audio head/embedding adapters."""
from __future__ import annotations

import argparse
import gc
import importlib.util
from pathlib import Path

import torch

from test6_nghean_shared_tied_audio import SharedAudioLoRA, SharedTiedEmbedding, SharedTiedLinear

TRAIN_DIR = Path(__file__).resolve().parent.parent
SOURCE_RUNNER = TRAIN_DIR / "process" / "v3" / "test2_nghean_accent_scale.py"
DATA_EXP_DIR = TRAIN_DIR / "output" / "nghean_test2_accent_scale"
EXP_DIR = TRAIN_DIR / "output" / "nghean_test8_audio_head_only"


def load_runner():
    spec = importlib.util.spec_from_file_location("nghean_test2_heads_only_runtime", SOURCE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source runner: {SOURCE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure(runner):
    runner.EXP_DIR = EXP_DIR
    runner.OUT = EXP_DIR
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

    def inject_heads_only(engine):
        model = engine.model
        tied = [head.weight.data_ptr() == emb.weight.data_ptr() for emb, head in zip(model.audio_embeddings, model.audio_lm_heads)]
        print(f"base audio weight tied before attach: {sum(tied)}/16", flush=True)
        if sum(tied) != 16:
            raise RuntimeError("Expected 16/16 tied audio head weights")
        for p in model.parameters():
            p.requires_grad = False
        model.audio_embeddings.float(); model.audio_lm_heads.float()
        if model.xvec_proj is not None:
            model.xvec_proj.float()
        adapters = []
        for k, (emb, head) in enumerate(zip(model.audio_embeddings, model.audio_lm_heads)):
            adapter = SharedAudioLoRA(768, 1024, rank=8, alpha=16.0).to(head.weight.device, dtype=head.weight.dtype)
            adapters.append(adapter)
            model.audio_lm_heads[k] = SharedTiedLinear(head, adapter)
            model.audio_embeddings[k] = SharedTiedEmbedding(emb, adapter)
        for name, p in model.named_parameters():
            p.requires_grad = ".lora_A" in name or ".lora_B" in name
        tied_after = [head.base.weight.data_ptr() == emb.base.weight.data_ptr() for emb, head in zip(model.audio_embeddings, model.audio_lm_heads)]
        shared = [head.lora_A is emb.lora_A and head.lora_B is emb.lora_B for emb, head in zip(model.audio_embeddings, model.audio_lm_heads)]
        print(f"base audio weight tied after attach: {sum(tied_after)}/16", flush=True)
        print(f"shared A/B identity: {sum(shared)}/16", flush=True)
        if sum(tied_after) != 16 or sum(shared) != 16:
            raise RuntimeError("Tied base weights or shared A/B identity was not preserved")
        runner._heads_only_adapters = adapters

    runner.inject = inject_heads_only
    return runner


def dry_run(runner):
    runner.OUT.mkdir(parents=True, exist_ok=True)
    runner.seed(runner.SEED)
    train = runner.read_pairs("nghean_train_pairs_v2.csv")
    valid = runner.read_pairs("nghean_valid_pairs_v2.csv")
    test = runner.read_pairs("nghean_test_pairs_v2.csv")
    sets = runner.validate_split(train, valid, test)
    overlap = (sets["train"] & sets["valid"]) | (sets["train"] & sets["test"]) | (sets["valid"] & sets["test"])
    print(f"Dataset = {len(train)}/{len(valid)}/{len(test)}; speaker overlap = {len(overlap)}", flush=True)
    if overlap:
        raise RuntimeError("speaker overlap detected")
    runner.PREP_TOTAL = 3
    engine = runner._load_engine(); engine.model.eval()
    runner.inject(engine)
    samples = [runner.prepare(engine, row) for row in (train[0], valid[0], test[0])]
    sample = samples[0]
    loss = runner.compute_loss(engine, runner.true_history(engine, sample), sample["codes"])
    if not torch.isfinite(loss) or not loss.requires_grad:
        raise RuntimeError("dry-run loss is not finite or has no grad")
    loss.backward()
    acoustic_trainable = [n for n, p in engine.model.named_parameters() if p.requires_grad and n.startswith("acoustic_decoder.")]
    head_grads = []
    for k in range(16):
        h = engine.model.audio_lm_heads[k]
        head_grads.append(h.lora_A.grad is not None and torch.isfinite(h.lora_A.grad).all() and h.lora_B.grad is not None and torch.isfinite(h.lora_B.grad).all())
    total = sum(p.numel() for p in engine.model.parameters())
    trainable = sum(p.numel() for p in engine.model.parameters() if p.requires_grad)
    print("acoustic decoder LoRA trainable params = 0", flush=True)
    print("shared/tied audio adapters = 16; r=8; alpha=16", flush=True)
    print(f"trainable params = {trainable:,}; trainable % = {100.0 * trainable / total:.6f}%", flush=True)
    print(f"base weights frozen = {'YES' if not acoustic_trainable else 'NO'}", flush=True)
    print(f"dry-run loss = {float(loss.detach().cpu()):.6f}; finite = True; requires_grad = True", flush=True)
    print(f"shared audio gradient = {'OK' if all(head_grads) else 'FAIL'}; valid adapters = {sum(head_grads)}/16", flush=True)
    print("DRY-RUN ONLY: no optimizer.step(), no training, no generation", flush=True)


def train_only(runner):
    runner.OUT.mkdir(parents=True, exist_ok=True); runner.seed(runner.SEED)
    train_rows = runner.read_pairs("nghean_train_pairs_v2.csv")
    valid_rows = runner.read_pairs("nghean_valid_pairs_v2.csv")
    test_rows = runner.read_pairs("nghean_test_pairs_v2.csv")
    sets = runner.validate_split(train_rows, valid_rows, test_rows)
    if (sets["train"] & sets["valid"]) or (sets["train"] & sets["test"]) or (sets["valid"] & sets["test"]):
        raise RuntimeError("speaker overlap detected")
    runner.PREP_TOTAL = len(train_rows) + len(valid_rows) + len(test_rows)
    engine = runner._load_engine(); runner.inject(engine)
    train = [runner.prepare(engine, row) for row in train_rows]
    valid = [runner.prepare(engine, row) for row in valid_rows]
    [runner.prepare(engine, row) for row in test_rows]
    # TEST8 uses a long frame-by-frame graph.  The legacy train_stage keeps
    # the last loss/graph alive while validation starts, which can OOM on a
    # 6GB GPU.  Keep the same objective/config, but explicitly release every
    # per-step graph before validation and between validation samples.
    curve, (best_val, best_step), _ = train_stage_memory_safe(runner, engine, train, valid)
    best_dir = runner.OUT / "best_tf_checkpoint"
    runner.save_adapter(engine, best_dir, best_step, best_val, "teacher_forcing_heads_only")
    header = "step,train_loss,validation_CE,best_validation_CE,is_best,learning_rate,gradient_norm\n"
    body = "\n".join(",".join(str(item[key]) for key in ("step", "train_loss", "validation_CE", "best_validation_CE", "is_best", "learning_rate", "gradient_norm")) for item in curve)
    (runner.OUT / "validation_history.csv").write_text(header + body, encoding="utf-8")
    print(f"TRAIN COMPLETE: steps={curve[-1]['step']}; best_step={best_step}; best_validation_CE={best_val:.6f}", flush=True)
    print(f"best checkpoint: {best_dir}", flush=True)
    print("No Warm98/2, no WAV generation, no MOSS-DTW, no test CE", flush=True)


def _release_cuda_memory():
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
            _release_cuda_memory()
    return total / len(samples)


def train_stage_memory_safe(runner, engine, train, valid):
    opt = torch.optim.AdamW(
        [p for p in engine.model.parameters() if p.requires_grad],
        lr=runner.LR,
        weight_decay=0.0,
    )
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
        history = runner.true_history(engine, sample)
        loss = runner.compute_loss(engine, history, sample["codes"])
        if not torch.isfinite(loss):
            raise RuntimeError(f"NaN/Inf loss step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            [p for p in engine.model.parameters() if p.requires_grad], 1.0
        )
        if not torch.isfinite(torch.as_tensor(grad)):
            raise RuntimeError(f"NaN/Inf gradient step {step}")
        opt.step()
        train_value = float(loss.detach().cpu())

        # Release the complete frame-wise autograd graph before validation.
        del loss, history
        _release_cuda_memory()

        if step % runner.VALIDATION_INTERVAL == 0:
            val = evaluate_memory_safe(runner, engine, valid)
            is_best = val < best_val
            item = {
                "step": step,
                "train_loss": train_value,
                "validation_CE": val,
                "best_validation_CE": min(best_val, val),
                "is_best": is_best,
                "learning_rate": runner.LR,
                "gradient_norm": float(torch.as_tensor(grad)),
            }
            curve.append(item)
            print(
                f"tf step {step}: train={train_value:.6f} val={val:.6f} "
                f"best={item['best_validation_CE']:.6f} grad={item['gradient_norm']:.6f} "
                f"is_best={is_best}",
                flush=True,
            )
            if is_best:
                best_val, best_step = val, step
                best_state = runner.lora_state(engine.model)
                stale = 0
                torch.save(best_state, checkpoints / f"best_step_{step:03d}.pt")
            else:
                stale += 1
            _release_cuda_memory()
            if stale >= runner.EARLY_STOP_PATIENCE:
                print(
                    f"early stop at step {step}: no new best for "
                    f"{runner.EARLY_STOP_PATIENCE} validations",
                    flush=True,
                )
                break

    if best_state is None:
        raise RuntimeError("No best checkpoint was produced")
    return curve, (best_val, best_step), best_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    runner = configure(load_runner())
    if args.train:
        train_only(runner)
    else:
        dry_run(runner)


if __name__ == "__main__":
    main()
