"""TEST 11: warm-start mixed-history curriculum from TF-LoRA step 100."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

TRAIN_DIR = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = TRAIN_DIR / "output" / "test11_warm_start_curriculum"
SOURCE_SRC = TRAIN_DIR.parent / "source_code" / "audio_model" / "src"
sys.path.insert(0, str(SOURCE_SRC))
sys.path.insert(0, str(TRAIN_DIR))

from lora_one_sample_ffn_test import inject_ffn_lora  # noqa: E402
from test10_mixed_history_pilot import (  # noqa: E402
    SEED, evaluate, free_run_metrics, load_engine, load_saved_lora, load_split,
    mixed_history_hs, prepare_sample, lora_state,
)

BRANCHES = {"warm_98_true_2_generated": 0.02, "warm_95_true_5_generated": 0.05}
STEPS = 30
LR = 5e-5


def run_branch(name: str, probability: float):
    torch.manual_seed(SEED)
    engine = load_engine()
    source_adapter = TRAIN_DIR / "output" / "test10_mixed_history" / "teacher_forcing_100"
    load_saved_lora(engine, source_adapter)
    train_rows, val_rows = load_split()
    train = [prepare_sample(engine, row) for row in train_rows]
    val = [prepare_sample(engine, row) for row in val_rows]
    baseline_val = evaluate(engine, val)
    optimizer = torch.optim.AdamW([p for p in engine.model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    best_val = baseline_val
    best_step = 0
    best_state = lora_state(engine.model)
    curve = [{"step": 0, "val_ce": baseline_val, "grad_norm": 0.0}]
    start = time.perf_counter()
    print(f"\n## {name}: probability_generated={probability}")
    print(f"warm-start adapter: {source_adapter}; source step=100; val baseline={baseline_val:.6f}")
    for step in range(1, STEPS + 1):
        sample = train[(step - 1) % len(train)]
        optimizer.zero_grad(set_to_none=True)
        hs = mixed_history_hs(engine, sample, probability, SEED + step)
        loss = __import__("overfit_one_sample_test").compute_loss(engine, hs, sample["codes"])
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"NaN/Inf loss at {name} step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in engine.model.parameters() if p.requires_grad], 1.0)
        if not torch.isfinite(torch.as_tensor(grad_norm)).item():
            raise RuntimeError(f"NaN/Inf gradient at {name} step {step}")
        optimizer.step()
        if step % 10 == 0 or step == STEPS:
            val_loss = evaluate(engine, val)
            item = {"step": step, "val_ce": val_loss, "grad_norm": float(torch.as_tensor(grad_norm).cpu()), "elapsed_sec": time.perf_counter() - start}
            curve.append(item)
            print(f"step {step}: val={val_loss:.6f}; grad_norm={item['grad_norm']:.6f}; elapsed={item['elapsed_sec']:.1f}s", flush=True)
            if val_loss < best_val:
                best_val, best_step = val_loss, step
                best_state = lora_state(engine.model)
    mode_dir = OUTPUT_ROOT / name
    mode_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, mode_dir / "adapter_model.pt")
    (mode_dir / "adapter_config.json").write_text(json.dumps({"rank": 8, "alpha": 16, "dropout": 0.0, "best_step": best_step, "best_validation_loss": best_val, "warm_start": "TF-LoRA step 100", "probability_generated": probability, "lr": LR, "steps": STEPS}, ensure_ascii=False, indent=2), encoding="utf-8")
    # Evaluate the best saved state, not necessarily the final state.
    engine.model.load_state_dict(best_state, strict=False)
    free_run = {}
    for sample in val:
        speaker = sample["row"]["speakerID"]
        free_run[speaker] = {}
        for sid, text in (("sentence_02", "Chiều nay chúng ta sẽ gặp nhau ở đâu?"), ("sentence_03", "Tôi vừa hoàn thành công việc và đang chuẩn bị về nhà.")):
            free_run[speaker][sid] = free_run_metrics(engine, sample, text)
    return {"mode": name, "probability_generated": probability, "warm_start_step": 100, "steps": STEPS, "lr": LR, "val_baseline": baseline_val, "best_step": best_step, "best_val_ce": best_val, "val_change": best_val - baseline_val, "curve": curve, "free_run": free_run, "no_nan_inf": True, "output": str(mode_dir)}


def main():
    results = {}
    for name, probability in BRANCHES.items():
        results[name] = run_branch(name, probability)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n## TEST 11 final")
    for name, result in results.items():
        print(f"{name}: best_step={result['best_step']}; val={result['best_val_ce']:.6f}; delta={result['val_change']:+.6f}")
    print(f"summary: {OUTPUT_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()
