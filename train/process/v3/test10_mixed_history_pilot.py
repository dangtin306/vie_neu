"""TEST 10: compare 100% true history with 90/10 mixed history.

The only training change is how the previous acoustic frame is selected for the
semantic backbone. The per-frame acoustic objective remains teacher-forced on
the true 16-code target. No upstream source is modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

TRAIN_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = TRAIN_DIR.parent
SOURCE_SRC = PROJECT_DIR / "source_code" / "audio_model" / "src"
CANDIDATES_DIR = PROJECT_DIR / "vimd_hp_candidates"
PREFERRED_CSV = CANDIDATES_DIR / "metadata_preferred_6_15s.csv"
ADAPTER_ROOT = TRAIN_DIR / "output" / "test5_multispeaker_lora"
OUTPUT_ROOT = TRAIN_DIR / "output" / "test10_mixed_history"
CHECKPOINT = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
MODEL_SUBFOLDER = "update"
MOSS_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
TRAIN_COUNT = 6
VAL_COUNT = 2
MAX_STEPS = 300
SEED = 20260827
SPEECH_TEMPERATURE = 0.8
SPEECH_TOP_K = 25
SPEECH_TOP_P = 0.95
SPEECH_REPETITION_PENALTY = 1.2
MAX_NEW_FRAMES = 300

sys.path.insert(0, str(SOURCE_SRC))
sys.path.insert(0, str(TRAIN_DIR))

from lora_one_sample_ffn_test import inject_ffn_lora  # noqa: E402
from overfit_one_sample_test import build_prompt, compute_loss, load_engine, precompute_h  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split():
    with PREFERRED_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if (CANDIDATES_DIR / r.get("downloaded_file", "")).is_file()]
    decisions = {(r.get("decision") or "").strip().casefold() for r in rows}
    if any(decisions - {""}):
        rows = [r for r in rows if (r.get("decision") or "").strip().casefold() == "giữ"]
    unique = {}
    for row in rows:
        unique.setdefault(row["speakerID"], row)
    # Match TEST 5 exactly: candidate_042 is the first training sample.
    speakers = sorted(unique, key=lambda s: (0 if unique[s].get("downloaded_file") == "candidate_042.wav" else 1, unique[s].get("downloaded_file", "")))
    if len(speakers) < TRAIN_COUNT + VAL_COUNT:
        raise RuntimeError(f"Không đủ speaker disjoint: {len(speakers)}")
    train = [unique[s] for s in speakers[:TRAIN_COUNT]]
    val = [unique[s] for s in speakers[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT]]
    if set(r["speakerID"] for r in train) & set(r["speakerID"] for r in val):
        raise RuntimeError("Train/validation speaker overlap")
    return train, val


def prepare_sample(engine, row):
    path = CANDIDATES_DIR / row["downloaded_file"]
    text = (row.get("transcript") or "").strip()
    if not text:
        raise RuntimeError(f"Transcript rỗng: {path.name}")
    speaker_emb, ref_codes = engine.prepare_reference(str(path), denoise=False, use_ref_codes=True)
    wav, sr = engine._load_mono(str(path), None)
    codes = torch.as_tensor(engine._encode_ref_wav(wav, sr), dtype=torch.long, device=engine.device)
    prompt = build_prompt(engine, text, ref_codes)
    true_hs = precompute_h(engine, prompt, codes, speaker_emb)
    return {"row": row, "path": path, "speaker_emb": speaker_emb, "ref_codes": np.asarray(ref_codes), "codes": codes, "prompt": prompt, "true_hs": true_hs}


@torch.no_grad()
def mixed_history_hs(engine, sample, generated_probability: float, seed: int):
    """Build semantic h_t with true/generated previous frames, without graph."""
    set_seed(seed)
    model = engine.model
    codes = sample["codes"]
    spk = engine._resolve_speaker_emb(sample["speaker_emb"])
    ids = sample["prompt"].unsqueeze(0).to(engine.device)
    semantic_dtype = next(model.semantic_backbone.parameters()).dtype
    embeds = model._build_inputs_embeds(ids, speaker_emb=spk)
    out = model.semantic_backbone(inputs_embeds=embeds.to(dtype=semantic_dtype), use_cache=True, return_dict=True)
    past = out.past_key_values
    h = out.last_hidden_state[:, -1]
    hs = []
    for t in range(len(codes)):
        hs.append(h.detach())
        if t + 1 >= len(codes):
            break
        if random.random() < generated_probability:
            generated, _ = model.decode_one_frame(h, text_token_id=torch.tensor([engine.config.speech_generation_start_token_id], device=engine.device), temperature=SPEECH_TEMPERATURE, top_k=SPEECH_TOP_K, audio_top_p=SPEECH_TOP_P, repetition_penalty=SPEECH_REPETITION_PENALTY)
            previous = generated.detach()
        else:
            previous = codes[t]
        row = torch.full((1, 1, engine.config.n_vq + 1), engine.config.audio_pad_token_id, dtype=torch.long, device=engine.device)
        row[:, :, 0] = engine.config.speech_generation_start_token_id
        row[:, 0, 1:] = previous
        emb = model._build_inputs_embeds(row, speaker_emb=spk)
        out = model.semantic_backbone(inputs_embeds=emb.to(dtype=semantic_dtype), past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        h = out.last_hidden_state[:, 0]
    return hs


def lora_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if ".lora_A" in name or ".lora_B" in name}


def inject_and_freeze(engine):
    model = engine.model
    for p in model.parameters():
        p.requires_grad = False
    model.acoustic_decoder.float()
    model.audio_embeddings.float()
    model.audio_lm_heads.float()
    if model.xvec_proj is not None:
        model.xvec_proj.float()
    targets = inject_ffn_lora(model.acoustic_decoder.layers[0], rank=8, alpha=16.0, dropout=0.0)
    for name, p in model.named_parameters():
        p.requires_grad = ".lora_A" in name or ".lora_B" in name
    return targets


def evaluate(engine, samples):
    with torch.no_grad():
        values = [compute_loss(engine, sample["true_hs"], sample["codes"]) for sample in samples]
    return float(torch.stack(values).mean().cpu())


def save_adapter(engine, output_dir, step, val_loss, train_loss, train_speakers, val_speakers):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state(engine.model), output_dir / "adapter_model.pt")
    (output_dir / "adapter_config.json").write_text(json.dumps({"rank": 8, "alpha": 16, "dropout": 0.0, "best_step": step, "best_validation_loss": val_loss, "train_loss_at_best": train_loss, "train_speakers": train_speakers, "validation_speakers": val_speakers, "history_mode": "mixed_90_true_10_generated"}, ensure_ascii=False, indent=2), encoding="utf-8")


def free_run_metrics(engine, sample, text):
    from test8_free_running_trace import trace_generation
    from test7_pause_rhythm_diagnosis import waveform_metrics
    codes, trace, stop = trace_generation(engine, text, (sample["speaker_emb"], sample["ref_codes"]), SEED)
    wav = np.asarray(engine._decode_codes(codes), dtype=np.float32).reshape(-1)
    output = {"frames": int(len(codes)), "duration_sec": len(wav) / 48000.0, "stop": stop, "eos": stop == "eos", "hit_max": stop == "max_new_frames"}
    output.update({key: value for key, value in waveform_metrics(wav, codes).items() if key not in {"duration_sec", "generated_frames"}})
    return output


def load_saved_lora(engine, adapter_dir):
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    inject_and_freeze(engine)
    state = torch.load(adapter_dir / "adapter_model.pt", map_location="cpu", weights_only=True)
    expected = {name for name, _ in engine.model.named_parameters() if ".lora_A" in name or ".lora_B" in name}
    if set(state) != expected:
        raise RuntimeError(f"Saved adapter keys mismatch: missing={expected-set(state)}, unexpected={set(state)-expected}")
    engine.model.load_state_dict(state, strict=False)
    return config


def existing_teacher_result():
    """Reuse the completed 100-step TF baseline instead of retraining it."""
    adapter_dir = OUTPUT_ROOT / "teacher_forcing_100"
    if not (adapter_dir / "adapter_model.pt").is_file():
        raise RuntimeError("Thiếu teacher-forcing artifact 100-step để reuse.")
    engine = load_engine()
    config = load_saved_lora(engine, adapter_dir)
    _, val_rows = load_split()
    val = [prepare_sample(engine, r) for r in val_rows]
    free_run = {}
    for sample in val:
        speaker = sample["row"]["speakerID"]
        free_run[speaker] = {}
        for sentence_id, text in (("sentence_02", "Chiều nay chúng ta sẽ gặp nhau ở đâu?"), ("sentence_03", "Tôi vừa hoàn thành công việc và đang chuẩn bị về nhà.")):
            free_run[speaker][sentence_id] = free_run_metrics(engine, sample, text)
    return {"mode": "teacher_forcing_100", "train_baseline": 5.782915, "val_baseline": 5.647254, "best_step": int(config.get("best_step", 100)), "best_train_ce": float(config.get("train_loss_at_best", 5.633272)), "best_val_ce": float(config.get("best_validation_loss", 5.570426)), "train_reduction": 5.782915 - float(config.get("train_loss_at_best", 5.633272)), "val_reduction": 5.647254 - float(config.get("best_validation_loss", 5.570426)), "free_run": free_run, "steps": 100, "reused_existing_artifact": True}


def evaluate_saved_mode(mode: str, adapter_dir: Path, training_result: dict):
    engine = load_engine()
    load_saved_lora(engine, adapter_dir)
    _, val_rows = load_split()
    val = [prepare_sample(engine, r) for r in val_rows]
    free_run = {}
    for sample in val:
        speaker = sample["row"]["speakerID"]
        free_run[speaker] = {}
        for sentence_id, text in (("sentence_02", "Chiều nay chúng ta sẽ gặp nhau ở đâu?"), ("sentence_03", "Tôi vừa hoàn thành công việc và đang chuẩn bị về nhà.")):
            free_run[speaker][sentence_id] = free_run_metrics(engine, sample, text)
    result = dict(training_result)
    result["free_run"] = free_run
    result["mode"] = mode
    return result


def run_mode(mode: str, steps: int):
    set_seed(SEED)
    engine = load_engine()
    targets = inject_and_freeze(engine)
    train_rows, val_rows = load_split()
    train = [prepare_sample(engine, r) for r in train_rows]
    val = [prepare_sample(engine, r) for r in val_rows]
    train_base = evaluate(engine, train)
    val_base = evaluate(engine, val)
    optimizer = torch.optim.AdamW([p for p in engine.model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0)
    mode_dir = OUTPUT_ROOT / mode
    best_val = val_base
    best_step = 0
    best_train = train_base
    curve = [{"step": 0, "train_ce": train_base, "val_ce": val_base, "grad_norm": 0.0}]
    print(f"\n## {mode}")
    print(f"targets: {targets}")
    print(f"train samples={len(train)}; validation samples={len(val)}; train baseline={train_base:.6f}; val baseline={val_base:.6f}")
    start = time.perf_counter()
    for step in range(1, steps + 1):
        sample = train[(step - 1) % len(train)]
        optimizer.zero_grad(set_to_none=True)
        hs = sample["true_hs"] if mode == "teacher_forcing_100" else mixed_history_hs(engine, sample, 0.10, SEED + step)
        loss = compute_loss(engine, hs, sample["codes"])
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"NaN/Inf loss at {mode} step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in engine.model.parameters() if p.requires_grad], 1.0)
        if not torch.isfinite(torch.as_tensor(grad_norm)).item():
            raise RuntimeError(f"NaN/Inf gradient at {mode} step {step}")
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == steps:
            train_loss = evaluate(engine, train)
            val_loss = evaluate(engine, val)
            row = {"step": step, "train_ce": train_loss, "val_ce": val_loss, "grad_norm": float(torch.as_tensor(grad_norm).cpu()), "elapsed_sec": time.perf_counter() - start}
            curve.append(row)
            print(f"step {step}: train={train_loss:.6f}; val={val_loss:.6f}; grad_norm={row['grad_norm']:.6f}; elapsed={row['elapsed_sec']:.1f}s", flush=True)
            if val_loss < best_val:
                best_val, best_step, best_train = val_loss, step, train_loss
                save_adapter(engine, mode_dir, step, val_loss, train_loss, [r["speakerID"] for r in train_rows], [r["speakerID"] for r in val_rows])
    if not (mode_dir / "adapter_model.pt").exists():
        save_adapter(engine, mode_dir, best_step, best_val, best_train, [r["speakerID"] for r in train_rows], [r["speakerID"] for r in val_rows])
    free_run = {}
    for sample in val:
        speaker = sample["row"]["speakerID"]
        free_run[speaker] = {}
        for sentence_id, text in (("sentence_02", "Chiều nay chúng ta sẽ gặp nhau ở đâu?"), ("sentence_03", "Tôi vừa hoàn thành công việc và đang chuẩn bị về nhà.")):
            free_run[speaker][sentence_id] = free_run_metrics(engine, sample, text)
    return {"mode": mode, "train_baseline": train_base, "val_baseline": val_base, "best_step": best_step, "best_train_ce": best_train, "best_val_ce": best_val, "train_reduction": train_base - best_train, "val_reduction": val_base - best_val, "curve": curve, "free_run": free_run, "train_speakers": [r["speakerID"] for r in train_rows], "val_speakers": [r["speakerID"] for r in val_rows], "steps": steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--eval-only", action="store_true", help="Chỉ đánh giá free-running từ adapter đã lưu")
    args = parser.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps phải >= 1")
    if args.eval_only:
        summary_path = OUTPUT_ROOT / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError("Thiếu summary TEST 10 để eval-only.")
        old = json.loads(summary_path.read_text(encoding="utf-8"))
        results = {"teacher_forcing_100": evaluate_saved_mode("teacher_forcing_100", OUTPUT_ROOT / "teacher_forcing_100", old["teacher_forcing_100"]), "mixed_90_true_10_generated": evaluate_saved_mode("mixed_90_true_10_generated", OUTPUT_ROOT / "mixed_90_true_10_generated", old["mixed_90_true_10_generated"])}
        summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"summary: {summary_path}")
        return
    results = {"teacher_forcing_100": existing_teacher_result()}
    results["mixed_90_true_10_generated"] = run_mode("mixed_90_true_10_generated", args.steps)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n## TEST 10 final")
    for mode, result in results.items():
        print(f"{mode}: best_step={result['best_step']}; best_val={result['best_val_ce']:.6f}; val_reduction={result['val_reduction']:.6f}")
    print(f"summary: {OUTPUT_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()
