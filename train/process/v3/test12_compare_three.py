"""Inference-only comparison for TEST12: TF, TEST11 98/2, TEST12 best."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import torch

TRAIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAIN_DIR))

from test10_mixed_history_pilot import load_split  # noqa: E402
from test12_warm98_longer import free_eval  # noqa: E402

MODELS = {
    "TF_LoRA_step100": TRAIN_DIR / "output" / "test10_mixed_history" / "teacher_forcing_100" / "adapter_model.pt",
    "TEST11_warm98_step30": TRAIN_DIR / "output" / "test11_warm_start_curriculum" / "warm_98_true_2_generated" / "adapter_model.pt",
    "TEST12_warm98_step040": TRAIN_DIR / "output" / "test12_warm98_longer" / "checkpoint_040" / "adapter_model.pt",
}
OUT = TRAIN_DIR / "output" / "test12_warm98_longer" / "compare_three_models.json"


def main() -> None:
    _, val_rows = load_split()
    all_results = {}
    for name, path in MODELS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"evaluating {name} ...", flush=True)
        state = torch.load(path, map_location="cpu", weights_only=True)
        all_results[name] = free_eval(state, val_rows)
        print(json.dumps(all_results[name]["aggregate"], ensure_ascii=False), flush=True)
    OUT.write_text(json.dumps({"models": all_results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary: {OUT}")


if __name__ == "__main__":
    main()
