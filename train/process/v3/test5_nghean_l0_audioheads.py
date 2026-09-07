"""Controlled Nghệ An pilot: layer-0 LoRA plus all 16 audio LM heads."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch


TRAIN_DIR = Path(__file__).resolve().parent.parent
SOURCE_RUNNER = TRAIN_DIR / "process" / "v3" / "test2_nghean_accent_scale.py"
DATA_EXP_DIR = TRAIN_DIR / "output" / "nghean_test2_accent_scale"
EXP_DIR = TRAIN_DIR / "output" / "nghean_test5_l0_audioheads"


def load_runner():
    spec = importlib.util.spec_from_file_location("nghean_test2_audioheads_runtime", SOURCE_RUNNER)
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

    from lora_one_sample_test import LoRALinear

    def inject_layer0_and_audio_heads(engine):
        model = engine.model
        tied_before = []
        for k, (embedding, head) in enumerate(zip(model.audio_embeddings, model.audio_lm_heads)):
            tied_before.append(
                head.weight is embedding.weight
                or head.weight.data_ptr() == embedding.weight.data_ptr()
            )
        print(f"audio heads tied before LoRA: {sum(tied_before)}/16", flush=True)
        if sum(tied_before) != 16:
            raise RuntimeError("Expected all 16 audio heads to be tied before LoRA attach")

        for parameter in model.parameters():
            parameter.requires_grad = False
        model.acoustic_decoder.float()
        model.audio_embeddings.float()
        model.audio_lm_heads.float()
        if model.xvec_proj is not None:
            model.xvec_proj.float()

        targets = list(runner.inject_ffn_lora(model.acoustic_decoder.layers[0], 8, 16.0, 0.0))
        for k, head in enumerate(model.audio_lm_heads):
            if not isinstance(head, torch.nn.Linear):
                raise TypeError(f"audio_lm_heads.{k} must be nn.Linear, got {type(head).__name__}")
            base = head
            model.audio_lm_heads[k] = LoRALinear(base, rank=8, alpha=16.0, dropout=0.0)
            targets.append(f"audio_lm_heads.{k}")
            print(
                f"runtime module audio_lm_heads.{k}: {base.in_features}->{base.out_features}, "
                f"params={sum(p.numel() for p in base.parameters())}",
                flush=True,
            )

        for name, parameter in model.named_parameters():
            parameter.requires_grad = ".lora_A" in name or ".lora_B" in name

        tied_after = []
        for k, (embedding, wrapped_head) in enumerate(zip(model.audio_embeddings, model.audio_lm_heads)):
            tied_after.append(
                wrapped_head.base.weight is embedding.weight
                or wrapped_head.base.weight.data_ptr() == embedding.weight.data_ptr()
            )
        print(f"base weight tying preserved after LoRA: {sum(tied_after)}/16", flush=True)
        if sum(tied_after) != 16:
            raise RuntimeError("LoRA attach did not preserve all audio base weight ties")

        runner._audiohead_targets = targets
        runner._audiohead_tied = sum(tied_after)

    runner.inject = inject_layer0_and_audio_heads
    return runner


def dry_run(runner):
    runner.OUT.mkdir(parents=True, exist_ok=True)
    runner.seed(runner.SEED)
    train = runner.read_pairs("nghean_train_pairs_v2.csv")
    valid = runner.read_pairs("nghean_valid_pairs_v2.csv")
    test = runner.read_pairs("nghean_test_pairs_v2.csv")
    sets = runner.validate_split(train, valid, test)
    overlap = bool((sets["train"] & sets["valid"]) or (sets["train"] & sets["test"]) or (sets["valid"] & sets["test"]))
    print(f"Dataset = {len(train)}/{len(valid)}/{len(test)}", flush=True)
    print(f"Speaker overlap = {len(sets['train'] & sets['valid']) + len(sets['train'] & sets['test']) + len(sets['valid'] & sets['test'])} (expected 0)", flush=True)
    if overlap:
        raise RuntimeError("speaker overlap detected")

    runner.PREP_TOTAL = 3
    engine = runner._load_engine()
    runner.inject(engine)
    samples = [runner.prepare(engine, row) for row in (train[0], valid[0], test[0])]
    sample = samples[0]
    loss = runner.compute_loss(engine, runner.true_history(engine, sample), sample["codes"])
    if not torch.isfinite(loss):
        raise RuntimeError(f"dry-run loss is not finite: {loss}")
    if not loss.requires_grad:
        raise RuntimeError("dry-run loss.requires_grad is False")
    loss.backward()

    model = engine.model
    layer_names = ["acoustic_decoder.layers.0"]
    layer_grads = [p for n, p in model.named_parameters() if n.startswith(layer_names[0]) and ".lora_" in n]
    head_grads = [p for n, p in model.named_parameters() if n.startswith("audio_lm_heads.") and ".lora_" in n]
    finite_layer = [p for p in layer_grads if p.grad is not None and torch.isfinite(p.grad).all()]
    finite_head = [p for p in head_grads if p.grad is not None and torch.isfinite(p.grad).all()]
    head_ok = all(
        any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for n, p in model.named_parameters()
            if n.startswith(f"audio_lm_heads.{k}.") and ".lora_" in n)
        for k in (0, 7, 15)
    )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"r = 8; alpha = 16; layer0 modules = 5; audio heads = 16; total targets = 21", flush=True)
    print(f"trainable params = {trainable:,}; trainable % = {100.0 * trainable / total:.6f}%", flush=True)
    print(f"dry-run loss = {float(loss.detach().cpu()):.6f}; finite = True; requires_grad = True", flush=True)
    print(f"layer0 gradient = {'OK' if len(finite_layer) == len(layer_grads) else 'FAIL'} (finite {len(finite_layer)}/{len(layer_grads)})", flush=True)
    print(f"audio head gradient = {'OK' if head_ok and len(finite_head) == len(head_grads) else 'FAIL'} (finite {len(finite_head)}/{len(head_grads)} params; heads 0/7/15 checked)", flush=True)
    print("DRY-RUN ONLY: no optimizer.step(), no training, no generation", flush=True)


def train_only(runner):
    """Run the TEST2 teacher-forcing loop only; never evaluate/generate WAV."""
    runner.OUT.mkdir(parents=True, exist_ok=True)
    runner.seed(runner.SEED)
    train_rows = runner.read_pairs("nghean_train_pairs_v2.csv")
    valid_rows = runner.read_pairs("nghean_valid_pairs_v2.csv")
    test_rows = runner.read_pairs("nghean_test_pairs_v2.csv")
    sets = runner.validate_split(train_rows, valid_rows, test_rows)
    if sets["train"] & sets["valid"] or sets["train"] & sets["test"] or sets["valid"] & sets["test"]:
        raise RuntimeError("speaker overlap detected")
    runner.PREP_TOTAL = len(train_rows) + len(valid_rows) + len(test_rows)
    print(f"dataset rows train/valid/test: {len(train_rows)}/{len(valid_rows)}/{len(test_rows)}", flush=True)
    engine = runner._load_engine()
    runner.inject(engine)
    train = [runner.prepare(engine, row) for row in train_rows]
    valid = [runner.prepare(engine, row) for row in valid_rows]
    # Prepare test rows for parity/checking, but do not evaluate or generate them.
    [runner.prepare(engine, row) for row in test_rows]
    curve, (best_val, best_step), _ = runner.train_stage(engine, train, valid)
    best_dir = runner.OUT / "best_tf_checkpoint"
    runner.save_adapter(engine, best_dir, best_step, best_val, "teacher_forcing")
    history = "step,train_loss,validation_CE,best_validation_CE,is_best,learning_rate,gradient_norm\n"
    history += "\n".join(
        ",".join(str(item[key]) for key in ("step", "train_loss", "validation_CE", "best_validation_CE", "is_best", "learning_rate", "gradient_norm"))
        for item in curve
    )
    (runner.OUT / "validation_history.csv").write_text(history, encoding="utf-8")
    print(f"TRAIN COMPLETE: steps={curve[-1]['step']}; best_step={best_step}; best_validation_CE={best_val:.6f}", flush=True)
    print(f"best checkpoint: {best_dir}", flush=True)
    print("No test CE, no Warm98/2, no WAV generation, no MOSS diagnostic", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="run teacher-forcing training only; no generation")
    args = parser.parse_args()
    runner = configure(load_runner())
    if args.train:
        train_only(runner)
    else:
        dry_run(runner)


if __name__ == "__main__":
    main()
