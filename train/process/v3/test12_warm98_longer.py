"""TEST 12: 98/2 warm-start for 100 steps, with checkpoint stability checks."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

TRAIN_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = TRAIN_DIR.parent
SOURCE_SRC = PROJECT_DIR / "source_code" / "audio_model" / "src"
OUT = TRAIN_DIR / "output" / "test12_warm98_longer"
TF_ADAPTER = TRAIN_DIR / "output" / "test10_mixed_history" / "teacher_forcing_100"
SPEAKERS = ["spk_15_0022", "spk_15_0025"]
SENTENCES = [
    ("sentence_01", "Hôm nay thời tiết khá dễ chịu."),
    ("sentence_02", "Chiều nay chúng ta sẽ gặp nhau ở đâu?"),
    ("sentence_03", "Tôi vừa hoàn thành công việc và đang chuẩn bị về nhà."),
    ("sentence_04", "Cậu đã ăn cơm chưa?"),
    ("sentence_06", "Mọi người đang chờ ở phía trước."),
]
STEPS = 100
LR = 5e-5
SEED = 20260827

sys.path.insert(0, str(SOURCE_SRC))
sys.path.insert(0, str(TRAIN_DIR))

from lora_one_sample_ffn_test import inject_ffn_lora  # noqa: E402
from overfit_one_sample_test import compute_loss  # noqa: E402
from test10_mixed_history_pilot import (  # noqa: E402
    evaluate, load_engine, load_saved_lora, load_split, mixed_history_hs, prepare_sample, lora_state,
)


def save_state(state, directory: Path, step: int, val: float):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(state, directory / "adapter_model.pt")
    (directory / "adapter_config.json").write_text(json.dumps({"rank": 8, "alpha": 16, "dropout": 0.0, "best_step": step, "validation_ce": val, "warm_start": "TF-LoRA step 100", "probability_generated": 0.02, "lr": LR}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_val_data(engine):
    _, val_rows = load_split()
    return [prepare_sample(engine, row) for row in val_rows]


def train():
    torch.manual_seed(SEED)
    engine = load_engine()
    load_saved_lora(engine, TF_ADAPTER)
    train_rows, val_rows = load_split()
    train = [prepare_sample(engine, row) for row in train_rows]
    val = [prepare_sample(engine, row) for row in val_rows]
    baseline = evaluate(engine, val)
    optimizer = torch.optim.AdamW([p for p in engine.model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    checkpoints = {}
    curve = [{"step": 0, "val_ce": baseline, "grad_norm": 0.0}]
    for step in range(1, STEPS + 1):
        sample = train[(step - 1) % len(train)]
        optimizer.zero_grad(set_to_none=True)
        hs = mixed_history_hs(engine, sample, 0.02, SEED + step)
        loss = compute_loss(engine, hs, sample["codes"])
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"NaN/Inf loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_([p for p in engine.model.parameters() if p.requires_grad], 1.0)
        if not torch.isfinite(torch.as_tensor(grad)).item():
            raise RuntimeError(f"NaN/Inf gradient at step {step}")
        optimizer.step()
        if step % 20 == 0 or step == STEPS:
            val_ce = evaluate(engine, val)
            state = lora_state(engine.model)
            checkpoints[step] = state
            save_state(state, OUT / f"checkpoint_{step:03d}", step, val_ce)
            item = {"step": step, "val_ce": val_ce, "grad_norm": float(torch.as_tensor(grad).cpu()), "elapsed_sec": time.perf_counter()}
            curve.append(item)
            print(f"checkpoint {step}: val_ce={val_ce:.6f}; grad_norm={item['grad_norm']:.6f}", flush=True)
    return baseline, curve, checkpoints, train_rows, val_rows


def free_eval(adapter_state, val_rows):
    from test7_pause_rhythm_diagnosis import waveform_metrics
    from test8_free_running_trace import trace_generation
    engine = load_engine()
    load_saved_lora(engine, TF_ADAPTER)
    engine.model.load_state_dict(adapter_state, strict=False)
    results = {}
    for row in val_rows:
        sample = prepare_sample(engine, row)
        speaker = row["speakerID"]
        results[speaker] = {}
        for sid, text in SENTENCES:
            codes, _, stop = trace_generation(engine, text, (sample["speaker_emb"], sample["ref_codes"]), SEED + int(sid[-2:]))
            wav = np.asarray(engine._decode_codes(codes), dtype=np.float32).reshape(-1)
            metric = waveform_metrics(wav, codes)
            results[speaker][sid] = {"frames": int(len(codes)), "duration_sec": len(wav) / 48000.0, "eos": stop == "eos", "hit_max": stop == "max_new_frames", "stop": stop, **{k: v for k, v in metric.items() if k not in {"duration_sec", "generated_frames"}}}
    flat = [item for speaker in results.values() for item in speaker.values()]
    return {"cases": results, "aggregate": {"duration_mean": float(np.mean([x["duration_sec"] for x in flat])), "frames_mean": float(np.mean([x["frames"] for x in flat])), "eos_success": int(sum(x["eos"] for x in flat)), "hit_max": int(sum(x["hit_max"] for x in flat)), "silence_ratio_mean": float(np.mean([x["silence_ratio"] for x in flat])), "longest_silence_mean": float(np.mean([x["longest_silence_sec"] for x in flat])), "repeated_frame_ratio_mean": float(np.mean([x["repeated_frame_ratio"] for x in flat])), "over_10s": int(sum(x["duration_sec"] > 10 for x in flat)), "over_15s": int(sum(x["duration_sec"] > 15 for x in flat)), "silence_over_500ms": int(sum(x["silence_over_500ms"] for x in flat))}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline, curve, checkpoints, train_rows, val_rows = train()
    # Check every saved checkpoint for free-running stability; selection requires
    # no max-frame case and then prefers lower true-history validation CE.
    free = {str(step): free_eval(state, val_rows) for step, state in checkpoints.items()}
    eligible = [step for step in checkpoints if free[str(step)]["aggregate"]["hit_max"] == 0 and free[str(step)]["aggregate"]["eos_success"] == 10]
    best_step = min(eligible, key=lambda step: next(x["val_ce"] for x in curve if x["step"] == step)) if eligible else min(checkpoints, key=lambda step: next(x["val_ce"] for x in curve if x["step"] == step))
    best = {"step": best_step, "val_ce": next(x["val_ce"] for x in curve if x["step"] == best_step), "free_run": free[str(best_step)]}
    (OUT / "summary.json").write_text(json.dumps({"config": {"steps": STEPS, "lr": LR, "probability_generated": 0.02, "warm_start_step": 100, "speakers": SPEAKERS, "sentences": [x[0] for x in SENTENCES]}, "tf_baseline_val_ce": baseline, "curve": curve, "checkpoint_free_run": free, "eligible_steps": eligible, "best": best, "train_speakers": [r["speakerID"] for r in train_rows], "validation_speakers": [r["speakerID"] for r in val_rows]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best checkpoint: step={best_step}; val_ce={best['val_ce']:.6f}; eligible={eligible}")
    print(f"summary: {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
