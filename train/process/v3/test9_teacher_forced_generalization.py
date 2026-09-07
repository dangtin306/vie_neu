"""TEST 9: Base vs LoRA on true-history teacher-forced unseen clips.

This deliberately does not free-run, train, backpropagate, or update weights.
It answers whether the TEST 5 LoRA improves the acoustic target distribution on
unseen speakers when both models receive identical true frame history.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TRAIN_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = TRAIN_DIR.parent
SOURCE_SRC = PROJECT_DIR / "source_code" / "audio_model" / "src"
CANDIDATES_DIR = PROJECT_DIR / "vimd_hp_candidates"
PREFERRED_CSV = CANDIDATES_DIR / "metadata_preferred_6_15s.csv"
ADAPTER_DIR = TRAIN_DIR / "output" / "test5_multispeaker_lora"
OUTPUT_DIR = TRAIN_DIR / "output" / "test9_teacher_forced_generalization"
BASE_CHECKPOINT = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
MODEL_SUBFOLDER = "update"
MOSS_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
SPEAKERS = ["spk_15_0022", "spk_15_0025"]

sys.path.insert(0, str(SOURCE_SRC))
sys.path.insert(0, str(TRAIN_DIR))

from lora_one_sample_ffn_test import inject_ffn_lora  # noqa: E402
from overfit_one_sample_test import precompute_h  # noqa: E402


def load_engine():
    from vieneu._v3_turbo_engine.inference_v3_turbo import VieNeuTTSv3Turbo

    return VieNeuTTSv3Turbo(checkpoint_path=BASE_CHECKPOINT, model_subfolder=MODEL_SUBFOLDER, moss_tokenizer_path=MOSS_REPO, device="auto", dtype="auto")


def load_rows():
    with PREFERRED_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    found = {row.get("speakerID"): row for row in rows if row.get("speakerID") in SPEAKERS}
    if set(found) != set(SPEAKERS):
        raise RuntimeError(f"Thiếu speaker: {set(SPEAKERS) - set(found)}")
    return found


def target_codes(engine, path: Path) -> torch.Tensor:
    wav, sr = engine._load_mono(str(path), None)
    return torch.as_tensor(engine._encode_ref_wav(wav, sr), dtype=torch.long, device=engine.device)


def load_adapter(engine):
    config = json.loads((ADAPTER_DIR / "adapter_config.json").read_text(encoding="utf-8"))
    if int(config.get("best_step", -1)) != 100:
        raise RuntimeError(f"Không phải best adapter step 100: {config.get('best_step')}")
    names = inject_ffn_lora(engine.model.acoustic_decoder.layers[0], int(config["rank"]), float(config["alpha"]), float(config["dropout"]))
    state = torch.load(ADAPTER_DIR / "adapter_model.pt", map_location="cpu", weights_only=True)
    expected = {name for name, _ in engine.model.named_parameters() if ".lora_A" in name or ".lora_B" in name}
    if set(state) != expected:
        raise RuntimeError(f"Adapter keys mismatch: missing={expected-set(state)}, unexpected={set(state)-expected}")
    _, unexpected = engine.model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected adapter keys: {unexpected}")
    return names


@torch.inference_mode()
def teacher_forced_stats(engine, hs, codes):
    model = engine.model
    cfg = engine.config
    local_dtype = next(model.acoustic_decoder.parameters()).dtype
    rows = []
    n = len(codes)
    for t, h in enumerate(hs):
        cond = h[0].to(dtype=local_dtype)
        start = model.text_embeddings(torch.tensor([cfg.speech_generation_start_token_id], device=engine.device))[0].to(dtype=local_dtype)
        tokens = [cond, start]
        for k in range(cfg.n_vq - 1):
            tokens.append(model.audio_embeddings[k](codes[t, k].view(1))[0].to(dtype=local_dtype))
        local_out = model.acoustic_decoder(torch.stack(tokens).unsqueeze(0))
        eos_probs = torch.softmax(model.text_lm_head(local_out[0, 0]).float(), dim=-1)
        per_code = []
        for k in range(cfg.n_vq):
            logits = model.audio_lm_heads[k](local_out[:, k + 1]).float()
            target = codes[t, k].view(1)
            probs = torch.softmax(logits, dim=-1)
            per_code.append({
                "ce": float(F.cross_entropy(logits, target).cpu()),
                "entropy": float((-probs * torch.log(probs.clamp_min(1e-12))).sum().cpu()),
                "top1_prob": float(probs.max().cpu()),
                "log_probs": torch.log_softmax(logits, dim=-1),
            })
        frame_ce = float(np.mean([x["ce"] for x in per_code]))
        region = "head" if t < n / 3 else "tail" if t >= 2 * n / 3 else "middle"
        rows.append({
            "frame": t, "region": region, "frame_ce": frame_ce,
            "code": [codes[t, k].item() for k in range(cfg.n_vq)],
            "ce": [x["ce"] for x in per_code],
            "entropy": [x["entropy"] for x in per_code],
            "top1_prob": [x["top1_prob"] for x in per_code],
            "eos_prob": float(eos_probs[cfg.speech_generation_end_token_id].cpu()),
            "log_probs": [x["log_probs"].cpu() for x in per_code],
        })
    return rows


def summarize_pair(base_rows, lora_rows, n_vq):
    base_ce = np.array([r["ce"] for r in base_rows])
    lora_ce = np.array([r["ce"] for r in lora_rows])
    base_ent = np.array([r["entropy"] for r in base_rows])
    lora_ent = np.array([r["entropy"] for r in lora_rows])
    base_top = np.array([r["top1_prob"] for r in base_rows])
    lora_top = np.array([r["top1_prob"] for r in lora_rows])
    base_eos = np.array([r["eos_prob"] for r in base_rows])
    lora_eos = np.array([r["eos_prob"] for r in lora_rows])
    kl = []
    for b, l in zip(base_rows, lora_rows):
        kl.append([float((torch.exp(b["log_probs"][k]) * (b["log_probs"][k] - l["log_probs"][k])).sum()) for k in range(n_vq)])
    def group(mask):
        return {
            "base_ce": float(base_ce[mask].mean()), "lora_ce": float(lora_ce[mask].mean()),
            "lora_minus_base_ce": float((lora_ce[mask] - base_ce[mask]).mean()),
            "lora_better_frame_percent": float((lora_ce[mask].mean(axis=1) < base_ce[mask].mean(axis=1)).mean() * 100),
            "base_entropy": float(base_ent[mask].mean()), "lora_entropy": float(lora_ent[mask].mean()),
            "base_top1": float(base_top[mask].mean()), "lora_top1": float(lora_top[mask].mean()),
            "base_eos_prob": float(base_eos[mask].mean()), "lora_eos_prob": float(lora_eos[mask].mean()),
            "mean_kl_base_to_lora": float(np.mean(np.asarray(kl)[mask])),
        }
    result = {"all": group(np.ones(len(base_rows), dtype=bool))}
    for name in ("head", "middle", "tail"):
        result[name] = group(np.array([r["region"] == name for r in base_rows]))
    result["codebook"] = []
    for k in range(n_vq):
        result["codebook"].append({
            "codebook": k, "base_ce": float(base_ce[:, k].mean()), "lora_ce": float(lora_ce[:, k].mean()),
            "lora_minus_base_ce": float((lora_ce[:, k] - base_ce[:, k]).mean()),
            "lora_better_percent": float((lora_ce[:, k] < base_ce[:, k]).mean() * 100),
            "base_entropy": float(base_ent[:, k].mean()), "lora_entropy": float(lora_ent[:, k].mean()),
            "base_top1": float(base_top[:, k].mean()), "lora_top1": float(lora_top[:, k].mean()),
        })
    result["lora_better_codebook_percent"] = float((lora_ce < base_ce).mean() * 100)
    return result, kl


def main():
    rows = load_rows()
    engine = load_engine()
    data = {}
    print("## TEST 9 — true-history teacher-forced Base vs LoRA")
    print("Note: each unseen validation speaker has only 1 available clip in candidates; 2 clips/speaker are unavailable.")
    for speaker in SPEAKERS:
        path = CANDIDATES_DIR / rows[speaker]["downloaded_file"]
        speaker_emb, ref_codes_np = engine.prepare_reference(str(path), denoise=False, use_ref_codes=True)
        codes = target_codes(engine, path)
        transcript = rows[speaker].get("transcript", "")
        if not transcript:
            raise RuntimeError(f"Transcript rỗng cho {speaker}; không được tự đoán text.")
        prompt = engine._build_prompt_2d(engine._resolve_phonemes(None, transcript), None, ref_codes_np, engine._resolve_style_id())
        hs = precompute_h(engine, prompt, codes, speaker_emb)
        data[speaker] = {"path": path, "codes": codes, "hs": hs, "speaker_emb": speaker_emb, "transcript": transcript}
        print(f"{speaker}: {path.name}; target_shape={tuple(codes.shape)}; transcript_chars={len(transcript)}")
    base_stats = {}
    for speaker in SPEAKERS:
        base_stats[speaker] = teacher_forced_stats(engine, data[speaker]["hs"], data[speaker]["codes"])
    load_adapter(engine)
    lora_stats = {speaker: teacher_forced_stats(engine, data[speaker]["hs"], data[speaker]["codes"]) for speaker in SPEAKERS}
    output = {"speakers": {}, "overall": None}
    all_base, all_lora = [], []
    for speaker in SPEAKERS:
        result, kl = summarize_pair(base_stats[speaker], lora_stats[speaker], engine.config.n_vq)
        output["speakers"][speaker] = {"file": data[speaker]["path"].name, "target_shape": list(data[speaker]["codes"].shape), "summary": result}
        all_base.extend(base_stats[speaker]); all_lora.extend(lora_stats[speaker])
        print(f"{speaker}: Base CE={result['all']['base_ce']:.6f}; LoRA CE={result['all']['lora_ce']:.6f}; LoRA better frames={result['all']['lora_better_frame_percent']:.2f}%")
    output["overall"], _ = summarize_pair(all_base, all_lora, engine.config.n_vq)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trace_rows = []
    for speaker in SPEAKERS:
        for b, l in zip(base_stats[speaker], lora_stats[speaker]):
            for k in range(engine.config.n_vq):
                trace_rows.append({"speakerID": speaker, "frame": b["frame"], "region": b["region"], "codebook": k, "base_ce": b["ce"][k], "lora_ce": l["ce"][k], "base_entropy": b["entropy"][k], "lora_entropy": l["entropy"][k], "base_top1_prob": b["top1_prob"][k], "lora_top1_prob": l["top1_prob"][k], "base_eos_prob": b["eos_prob"], "lora_eos_prob": l["eos_prob"], "lora_better": l["ce"][k] < b["ce"][k]})
    with (OUTPUT_DIR / "frame_codebook_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0])); writer.writeheader(); writer.writerows(trace_rows)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"overall: Base CE={output['overall']['all']['base_ce']:.6f}; LoRA CE={output['overall']['all']['lora_ce']:.6f}; LoRA better frames={output['overall']['all']['lora_better_frame_percent']:.2f}%; LoRA better codebooks={output['overall']['lora_better_codebook_percent']:.2f}%")
    print(f"output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
