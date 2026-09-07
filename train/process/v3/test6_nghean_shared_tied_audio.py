"""TEST6: layer-0 LoRA plus shared tied LoRA for audio heads/embeddings."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


TRAIN_DIR = Path(__file__).resolve().parent.parent
SOURCE_RUNNER = TRAIN_DIR / "process" / "v3" / "test2_nghean_accent_scale.py"
DATA_EXP_DIR = TRAIN_DIR / "output" / "nghean_test2_accent_scale"
EXP_DIR = TRAIN_DIR / "output" / "nghean_test6_shared_tied_audio"


class SharedAudioLoRA(nn.Module):
    """One A/B delta shared by the output head and input embedding of a codebook."""

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty(self.rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self._cached_delta = None
        self._cached_versions = None

    def delta(self) -> torch.Tensor:
        # Reuse one differentiable delta across all frames in a loss call.
        # Optimizer updates increment Parameter._version, forcing refresh on
        # the next step without changing the shared A/B semantics.
        versions = (self.lora_A._version, self.lora_B._version, torch.is_grad_enabled())
        if self._cached_delta is None or self._cached_versions != versions:
            self._cached_delta = (self.lora_B @ self.lora_A) * self.scaling
            self._cached_versions = versions
        return self._cached_delta


class SharedTiedLinear(nn.Module):
    def __init__(self, base: nn.Linear, adapter: SharedAudioLoRA):
        super().__init__()
        self.base = base
        self.adapter = adapter

    @property
    def lora_A(self):
        return self.adapter.lora_A

    @property
    def lora_B(self):
        return self.adapter.lora_B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The v3 Turbo backbone may emit BF16 while the tied audio weights
        # are kept in FP32 for stable adapter training.  F.linear requires
        # matching dtypes, so follow the runtime input dtype without changing
        # the stored/tied base weights.
        weight = self.base.weight.to(dtype=x.dtype) + self.adapter.delta().to(dtype=x.dtype)
        bias = self.base.bias.to(dtype=x.dtype) if self.base.bias is not None else None
        return F.linear(x, weight, bias)


class SharedTiedEmbedding(nn.Module):
    def __init__(self, base: nn.Embedding, adapter: SharedAudioLoRA):
        super().__init__()
        self.base = base
        self.adapter = adapter

    @property
    def lora_A(self):
        return self.adapter.lora_A

    @property
    def lora_B(self):
        return self.adapter.lora_B

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            input_ids,
            self.base.weight + self.adapter.delta(),
            self.base.padding_idx,
            self.base.max_norm,
            self.base.norm_type,
            self.base.scale_grad_by_freq,
            self.base.sparse,
        )


def load_runner():
    spec = importlib.util.spec_from_file_location("nghean_test2_shared_audio_runtime", SOURCE_RUNNER)
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

    def inject_shared(engine):
        model = engine.model
        tied_before = []
        for k, (embedding, head) in enumerate(zip(model.audio_embeddings, model.audio_lm_heads)):
            tied_before.append(head.weight is embedding.weight or head.weight.data_ptr() == embedding.weight.data_ptr())
        print(f"base audio weight tied before attach: {sum(tied_before)}/16", flush=True)
        if sum(tied_before) != 16:
            raise RuntimeError("Expected audio_lm_heads and audio_embeddings to be tied 16/16")

        for parameter in model.parameters():
            parameter.requires_grad = False
        model.acoustic_decoder.float()
        model.audio_embeddings.float()
        model.audio_lm_heads.float()
        if model.xvec_proj is not None:
            model.xvec_proj.float()

        targets = list(runner.inject_ffn_lora(model.acoustic_decoder.layers[0], 8, 16.0, 0.0))
        adapters = []
        for k, (embedding, head) in enumerate(zip(model.audio_embeddings, model.audio_lm_heads)):
            if head.weight.data_ptr() != embedding.weight.data_ptr():
                raise RuntimeError(f"base tying changed before attach at codebook {k}")
            adapter = SharedAudioLoRA(768, 1024, rank=8, alpha=16.0).to(head.weight.device, dtype=head.weight.dtype)
            adapters.append(adapter)
            model.audio_lm_heads[k] = SharedTiedLinear(head, adapter)
            model.audio_embeddings[k] = SharedTiedEmbedding(embedding, adapter)
            targets.append(f"audio_lm_heads[{k}]<->audio_embeddings[{k}]")
            print(f"shared audio adapter {k}: delta [1024,768], rank=8, alpha=16", flush=True)

        for name, parameter in model.named_parameters():
            parameter.requires_grad = ".lora_A" in name or ".lora_B" in name

        tied_after = []
        shared_after = []
        for k, (embedding, head) in enumerate(zip(model.audio_embeddings, model.audio_lm_heads)):
            tied_after.append(head.base.weight.data_ptr() == embedding.base.weight.data_ptr())
            shared_after.append(head.adapter is embedding.adapter and head.lora_A is embedding.lora_A and head.lora_B is embedding.lora_B)
        print(f"base audio weight tied after attach: {sum(tied_after)}/16", flush=True)
        print(f"shared A/B identity after attach: {sum(shared_after)}/16", flush=True)
        if sum(tied_after) != 16 or sum(shared_after) != 16:
            raise RuntimeError("Shared tied adapter identity/weight tying was not preserved")
        runner._shared_targets = targets
        runner._shared_adapters = adapters

    runner.inject = inject_shared
    return runner


def dry_run(runner):
    runner.OUT.mkdir(parents=True, exist_ok=True)
    runner.seed(runner.SEED)
    train = runner.read_pairs("nghean_train_pairs_v2.csv")
    valid = runner.read_pairs("nghean_valid_pairs_v2.csv")
    test = runner.read_pairs("nghean_test_pairs_v2.csv")
    sets = runner.validate_split(train, valid, test)
    overlap = (sets["train"] & sets["valid"]) | (sets["train"] & sets["test"]) | (sets["valid"] & sets["test"])
    print(f"Dataset = {len(train)}/{len(valid)}/{len(test)}", flush=True)
    print(f"Speaker overlap = {len(overlap)}", flush=True)
    if overlap:
        raise RuntimeError("speaker overlap detected")

    runner.PREP_TOTAL = 3
    # Use one engine and one fixed hidden history for the before/after
    # comparison; this removes unrelated differences between two model loads.
    engine = runner._load_engine()
    engine.model.eval()
    # TEST5's controlled path evaluates acoustic modules in FP32. Set this
    # before the baseline capture as well, so zero-delta compares like-for-like.
    engine.model.acoustic_decoder.float()
    engine.model.audio_embeddings.float()
    engine.model.audio_lm_heads.float()
    if engine.model.xvec_proj is not None:
        engine.model.xvec_proj.float()
    sample = runner.prepare(engine, train[0])
    hs = runner.true_history(engine, sample)
    from overfit_one_sample_test import teacher_forced_logits
    with torch.no_grad():
        base_logits_by_frame = [teacher_forced_logits(engine, h, target_frame) for h, target_frame in zip(hs, sample["codes"])]
        base_loss = runner.compute_loss(engine, hs, sample["codes"])

    runner.inject(engine)
    # Prepare the other split examples only to exercise the normal pipeline.
    runner.prepare(engine, valid[0])
    runner.prepare(engine, test[0])
    with torch.no_grad():
        max_logit_diff = 0.0
        for h, target_frame, base_logits in zip(hs, sample["codes"], base_logits_by_frame):
            test_logits = teacher_forced_logits(engine, h, target_frame)
            for left, right in zip(base_logits, test_logits):
                max_logit_diff = max(max_logit_diff, float((left - right).abs().max().cpu()))
        zero_delta_loss = runner.compute_loss(engine, hs, sample["codes"])
    loss = runner.compute_loss(engine, hs, sample["codes"])
    if not torch.isfinite(loss) or not torch.isfinite(base_loss):
        raise RuntimeError(f"zero-delta loss is not finite: base={base_loss}, test6={loss}")
    if not loss.requires_grad:
        raise RuntimeError("dry-run loss.requires_grad is False")
    loss_diff = float((zero_delta_loss.detach() - base_loss.detach()).abs().cpu())
    loss.backward()

    model = engine.model
    layer_params = [p for n, p in model.named_parameters() if n.startswith("acoustic_decoder.layers.0") and ".lora_" in n]
    adapters = []
    for k in range(16):
        head = model.audio_lm_heads[k]
        emb = model.audio_embeddings[k]
        adapters.append((head, emb))
    layer_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer_params)
    adapter_ok = []
    for head, emb in adapters:
        adapter_ok.append(
            head.lora_A.grad is not None and torch.isfinite(head.lora_A.grad).all()
            and head.lora_B.grad is not None and torch.isfinite(head.lora_B.grad).all()
            and head.lora_A is emb.lora_A and head.lora_B is emb.lora_B
        )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    delta_checks = []
    for head, emb in adapters:
        delta_checks.append(torch.equal(head.adapter.delta(), emb.adapter.delta()))
    print("r = 8; alpha = 16", flush=True)
    print("Layer0 LoRA modules = 5; shared audio adapters = 16; total targets = 21", flush=True)
    print(f"base W tied = 16/16; shared delta used by head+embedding = {'YES' if all(delta_checks) else 'NO'}", flush=True)
    print(f"trainable params = {trainable:,}; trainable % = {100.0 * trainable / total:.6f}%", flush=True)
    print(f"TEST5 parameter match (333,824) = {'YES' if trainable == 333824 else 'NO'}", flush=True)
    print(f"initial delta W = 0: YES; max_abs_logit_diff = {max_logit_diff:.9f}", flush=True)
    print(f"zero-delta loss = {float(zero_delta_loss.detach().cpu()):.6f}; base loss = {float(base_loss.detach().cpu()):.6f}; abs diff = {loss_diff:.9f}", flush=True)
    print(f"zero-delta equivalence = {'PASS' if max_logit_diff <= 1e-5 and loss_diff <= 1e-5 else 'FAIL'}", flush=True)
    print(f"layer0 gradient = {'OK' if layer_ok else 'FAIL'}", flush=True)
    print(f"shared audio gradient = {'OK' if all(adapter_ok) else 'FAIL'}; valid adapters = {sum(adapter_ok)}/16", flush=True)
    print("separate embedding LoRA = NO", flush=True)
    print("DRY-RUN ONLY: no optimizer.step(), no training, no generation", flush=True)


def train_only(runner):
    """Run the controlled TEST2 teacher-forcing loop without inference output."""
    runner.OUT.mkdir(parents=True, exist_ok=True)
    runner.seed(runner.SEED)
    train_rows = runner.read_pairs("nghean_train_pairs_v2.csv")
    valid_rows = runner.read_pairs("nghean_valid_pairs_v2.csv")
    test_rows = runner.read_pairs("nghean_test_pairs_v2.csv")
    sets = runner.validate_split(train_rows, valid_rows, test_rows)
    overlap = (sets["train"] & sets["valid"]) | (sets["train"] & sets["test"]) | (sets["valid"] & sets["test"])
    if overlap:
        raise RuntimeError("speaker overlap detected")
    runner.PREP_TOTAL = len(train_rows) + len(valid_rows) + len(test_rows)
    print(f"dataset rows train/valid/test: {len(train_rows)}/{len(valid_rows)}/{len(test_rows)}", flush=True)
    engine = runner._load_engine()
    runner.inject(engine)
    train = [runner.prepare(engine, row) for row in train_rows]
    valid = [runner.prepare(engine, row) for row in valid_rows]
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
    print("No Warm98/2, no WAV generation, no MOSS-DTW, no test CE", flush=True)


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
