"""Nghệ An depth pilot: LoRA r=8 on acoustic-decoder layers 0 and 1.

This runner reuses the TEST2 implementation and dataset, changing only the
LoRA injection depth.  It intentionally does not run on import.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


TRAIN_DIR = Path(__file__).resolve().parent.parent
SOURCE_RUNNER = TRAIN_DIR / "process" / "v3" / "test2_nghean_accent_scale.py"
DATA_EXP_DIR = TRAIN_DIR / "output" / "nghean_test2_accent_scale"
EXP_DIR = TRAIN_DIR / "output" / "nghean_test4_depth_l01"


def _load_test2():
    spec = importlib.util.spec_from_file_location("nghean_test2_runtime", SOURCE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source runner: {SOURCE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    runner = _load_test2()

    # Keep the TEST2 dataset/split, while isolating all generated output/cache.
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

    def inject_depth_l01(engine):
        layer_count = len(engine.model.acoustic_decoder.layers)
        if layer_count < 2:
            raise RuntimeError(
                "TEST4 L0+L1 cannot run with this v3 Turbo checkpoint: "
                f"acoustic_decoder has {layer_count} layer(s), so layers.1 does not exist. "
                "No layer was added or substituted. Use a checkpoint configured with "
                "local_num_hidden_layers>=2 before running this depth experiment."
            )
        for parameter in engine.model.parameters():
            parameter.requires_grad = False
        engine.model.acoustic_decoder.float()
        engine.model.audio_embeddings.float()
        engine.model.audio_lm_heads.float()
        if engine.model.xvec_proj is not None:
            engine.model.xvec_proj.float()

        for layer_index in (0, 1):
            runner.inject_ffn_lora(
                engine.model.acoustic_decoder.layers[layer_index], 8, 16.0, 0.0
            )
        for name, parameter in engine.model.named_parameters():
            parameter.requires_grad = ".lora_A" in name or ".lora_B" in name

    runner.inject = inject_depth_l01
    runner.main()


if __name__ == "__main__":
    main()
