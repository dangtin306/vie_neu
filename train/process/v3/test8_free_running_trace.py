"""TEST 8: trace first Base/LoRA free-running divergence.

Inference-only diagnostic. It keeps the TEST 6 sampling configuration and uses
the same unseen speaker/reference for Base and LoRA. No parameters are trained.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

TRAIN_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = TRAIN_DIR.parent
SOURCE_SRC = PROJECT_DIR / "source_code" / "audio_model" / "src"
CANDIDATES_DIR = PROJECT_DIR / "vimd_hp_candidates"
PREFERRED_CSV = CANDIDATES_DIR / "metadata_preferred_6_15s.csv"
ADAPTER_DIR = TRAIN_DIR / "output" / "test5_multispeaker_lora"
OUTPUT_DIR = TRAIN_DIR / "output" / "test8_free_running_trace"
BASE_CHECKPOINT = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
MODEL_SUBFOLDER = "update"
MOSS_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
SPEAKER = "spk_15_0022"
SENTENCES = [
    ("sentence_02", "Chiều nay chúng ta sẽ gặp nhau ở đâu?"),
    ("sentence_03", "Tôi vừa hoàn thành công việc và đang chuẩn bị về nhà."),
]
TEMPERATURE = 0.8
TOP_K = 25
TOP_P = 0.95
REPETITION_PENALTY = 1.2
MAX_NEW_FRAMES = 300
SEED = 20260827
RMS_THRESHOLD = 0.01
SAMPLE_RATE = 48000

sys.path.insert(0, str(SOURCE_SRC))
sys.path.insert(0, str(TRAIN_DIR))

from lora_one_sample_ffn_test import inject_ffn_lora  # noqa: E402


def load_engine():
    from vieneu._v3_turbo_engine.inference_v3_turbo import VieNeuTTSv3Turbo

    return VieNeuTTSv3Turbo(checkpoint_path=BASE_CHECKPOINT, model_subfolder=MODEL_SUBFOLDER, moss_tokenizer_path=MOSS_REPO, device="auto", dtype="auto")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_row() -> dict[str, str]:
    with PREFERRED_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("speakerID") == SPEAKER:
            path = CANDIDATES_DIR / row.get("downloaded_file", "")
            if path.is_file():
                return row
    raise FileNotFoundError(f"Không tìm thấy reference cho {SPEAKER}")


def load_adapter(engine) -> list[str]:
    config_path = ADAPTER_DIR / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("best_step", -1)) != 100:
        raise RuntimeError(f"Adapter không phải best step 100: {config.get('best_step')}")
    names = inject_ffn_lora(engine.model.acoustic_decoder.layers[0], int(config["rank"]), float(config["alpha"]), float(config["dropout"]))
    state = torch.load(ADAPTER_DIR / "adapter_model.pt", map_location="cpu", weights_only=True)
    expected = {name for name, _ in engine.model.named_parameters() if ".lora_A" in name or ".lora_B" in name}
    if set(state) != expected:
        raise RuntimeError(f"Adapter keys mismatch: missing={expected-set(state)}, unexpected={set(state)-expected}")
    _, unexpected = engine.model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected adapter keys: {unexpected}")
    return names


@torch.no_grad()
def trace_generation(engine, text: str, reference_data, seed: int):
    from vieneu._v3_turbo_engine.modeling_v3_turbo import _sample_token
    from vieneu._v3_turbo_engine.rep_history import DEFAULT_REP_WINDOW, RepetitionHistory

    set_seed(seed)
    speaker_emb, ref_codes = reference_data
    phonemes = engine._resolve_phonemes(None, text)
    model = engine.model
    spk_t = engine._resolve_speaker_emb(speaker_emb)
    prompt = engine._build_prompt_2d(phonemes, None, ref_codes, engine._resolve_style_id())
    semantic_dtype = next(model.semantic_backbone.parameters()).dtype
    prefill_embeds = model._build_inputs_embeds(prompt.unsqueeze(0).to(engine.device), speaker_emb=spk_t)
    prefill = model.semantic_backbone(inputs_embeds=prefill_embeds.to(dtype=semantic_dtype), use_cache=True, return_dict=True)
    past_kv = prefill.past_key_values
    h = prefill.last_hidden_state[:, -1]
    hist = RepetitionHistory(engine.config.n_vq, DEFAULT_REP_WINDOW)
    rows = []
    frames = []
    eos_id = engine.config.speech_generation_end_token_id
    sgs_id = engine.config.speech_generation_start_token_id
    for frame_index in range(MAX_NEW_FRAMES):
        local_dtype = next(model.acoustic_decoder.parameters()).dtype
        cond = h[0].to(dtype=local_dtype)
        txt = model.text_embeddings(torch.tensor([sgs_id], device=engine.device))[0].to(dtype=local_dtype)
        hidden, pk, pv = model.acoustic_decoder.cached_step(torch.stack([cond, txt]).view(1, 2, -1), torch.tensor([0, 1], device=engine.device), [None] * len(model.acoustic_decoder.layers), [None] * len(model.acoustic_decoder.layers))
        local_out = hidden
        frame_codes = []
        code_stats = []
        for ch in range(engine.config.n_vq):
            vec = hidden[0, 1] if ch == 0 else hidden[0, 0]
            logits = model.audio_lm_heads[ch](vec).float()
            probs = torch.softmax(logits, dim=-1)
            entropy = float((-probs * torch.log(probs.clamp_min(1e-12))).sum().cpu())
            top1_prob = float(probs.max().cpu())
            selected = _sample_token(logits, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, repetition_penalty=REPETITION_PENALTY, prev_tokens=hist[ch])
            hist[ch].add(int(selected.item()))
            frame_codes.append(selected)
            code_stats.append((entropy, top1_prob, int(selected.item())))
            if ch + 1 < engine.config.n_vq:
                emb = model.audio_embeddings[ch](selected.unsqueeze(0))[0].to(dtype=local_dtype)
                hidden, pk, pv = model.acoustic_decoder.cached_step(emb.view(1, 1, -1), torch.tensor([ch + 2], device=engine.device), pk, pv)
        frame = torch.stack(frame_codes).cpu()
        frames.append(frame)
        text_head_dtype = next(model.text_lm_head.parameters()).dtype
        eos_probs = torch.softmax(model.text_lm_head(local_out[0, 0].to(dtype=text_head_dtype)).float(), dim=-1)
        eos_prob = float(eos_probs[eos_id].cpu())
        repeated = len(frames) > 1 and bool(torch.equal(frame, frames[-2]))
        rows.append({"frame": frame_index, "eos_prob": eos_prob, "mean_codebook_entropy": float(np.mean([x[0] for x in code_stats])), "mean_top1_prob": float(np.mean([x[1] for x in code_stats])), "codes": frame.tolist(), "repeated_frame": repeated, "codebook_entropy": [x[0] for x in code_stats], "codebook_top1_prob": [x[1] for x in code_stats], "selected_tokens": [x[2] for x in code_stats]})
        if int(model.text_lm_head(local_out[0, 0].to(dtype=text_head_dtype)).float().argmax().item()) == eos_id:
            return torch.stack(frames), rows, "eos"
        slot = torch.full((1, 1, engine.config.n_vq + 1), engine.config.audio_pad_token_id, dtype=torch.long, device=engine.device)
        engine._prepare_gen_slot_row(slot, frame_codes=frame.to(engine.device), sgs_id=sgs_id, audio_pad=engine.config.audio_pad_token_id)
        step_embeds = model._build_inputs_embeds(slot, speaker_emb=spk_t)
        step = model.semantic_backbone(inputs_embeds=step_embeds.to(dtype=semantic_dtype), past_key_values=past_kv, use_cache=True, return_dict=True)
        past_kv = step.past_key_values
        h = step.last_hidden_state[:, 0]
    return torch.stack(frames), rows, "max_new_frames"


def add_audio_silence(rows, wav, frame_count):
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    rms_values = []
    for row in rows:
        a = int(row["frame"] * len(wav) / max(frame_count, 1))
        b = int((row["frame"] + 1) * len(wav) / max(frame_count, 1))
        segment = wav[a:b]
        rms_values.append(float(np.sqrt(np.mean(segment * segment))) if len(segment) else 0.0)
    active = [i for i, value in enumerate(rms_values) if value >= RMS_THRESHOLD]
    first_active = active[0] if active else 0
    last_active = active[-1] if active else -1
    for row, rms in zip(rows, rms_values):
        row["frame_rms"] = rms
        # Ignore codec padding/silence before the first or after the last active frame.
        row["frame_silence"] = first_active <= row["frame"] <= last_active and rms < RMS_THRESHOLD


def main() -> None:
    row = load_row()
    engine = load_engine()
    ref_path = CANDIDATES_DIR / row["downloaded_file"]
    reference_data = engine.prepare_reference(str(ref_path), denoise=False, use_ref_codes=True)
    all_traces = {}
    summary = {}
    base_outputs = {}
    print("## TEST 8 — free-running Base/LoRA trace")
    print(f"speaker: {SPEAKER}; reference: {ref_path.name}; sentences: {[x[0] for x in SENTENCES]}")
    print(f"sampling: temperature={TEMPERATURE}, top_k={TOP_K}, top_p={TOP_P}, repetition_penalty={REPETITION_PENALTY}; seed base={SEED}")
    for mode in ("base", "lora"):
        if mode == "lora":
            names = load_adapter(engine)
            print(f"LoRA active: {len(names)} modules")
        for sentence_id, text in SENTENCES:
            seed = SEED + (2 if sentence_id == "sentence_02" else 3)
            codes, trace, stop = trace_generation(engine, text, reference_data, seed)
            wav = np.asarray(engine._decode_codes(codes), dtype=np.float32).reshape(-1)
            add_audio_silence(trace, wav, len(codes))
            out = OUTPUT_DIR / SPEAKER / f"{sentence_id}_{mode}_sampling.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out), wav, SAMPLE_RATE, subtype="PCM_16")
            all_traces[(mode, sentence_id)] = trace
            base_outputs[(mode, sentence_id)] = codes
            print(f"{mode} {sentence_id}: frames={len(codes)}, stop={stop}, duration={len(wav)/SAMPLE_RATE:.2f}s")
            summary[(mode, sentence_id)] = {"frames": len(codes), "stop": stop, "duration_sec": len(wav) / SAMPLE_RATE, "first_silence_frame": next((r["frame"] for r in trace if r["frame_silence"]), None), "first_repeat_frame": next((r["frame"] for r in trace if r["repeated_frame"]), None)}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trace_rows = []
    for sentence_id, _ in SENTENCES:
        base_codes = base_outputs[("base", sentence_id)]
        lora_codes = base_outputs[("lora", sentence_id)]
        max_len = max(len(base_codes), len(lora_codes))
        for frame in range(max_len):
            b = all_traces[("base", sentence_id)][frame] if frame < len(all_traces[("base", sentence_id)]) else None
            l = all_traces[("lora", sentence_id)][frame] if frame < len(all_traces[("lora", sentence_id)]) else None
            distance = int((base_codes[frame] != lora_codes[frame]).sum()) if frame < min(len(base_codes), len(lora_codes)) else None
            for mode, item in (("base", b), ("lora", l)):
                if item is None:
                    continue
                trace_rows.append({"model": mode, "sentence_id": sentence_id, "frame": frame, "eos_prob": item["eos_prob"], "mean_codebook_entropy": item["mean_codebook_entropy"], "mean_top1_prob": item["mean_top1_prob"], "repeated_frame": item["repeated_frame"], "frame_silence": item["frame_silence"], "frame_rms": item["frame_rms"], "code_distance_base_vs_lora": distance, "codes": json.dumps(item["codes"], ensure_ascii=False), "codebook_entropy": json.dumps(item["codebook_entropy"]), "codebook_top1_prob": json.dumps(item["codebook_top1_prob"]), "selected_tokens": json.dumps(item["selected_tokens"])} )
    fields = list(trace_rows[0])
    with (OUTPUT_DIR / "frame_trace.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trace_rows)
    divergences = {}
    for sentence_id, _ in SENTENCES:
        b = base_outputs[("base", sentence_id)]
        l = base_outputs[("lora", sentence_id)]
        distances = [(int((b[i] != l[i]).sum()), i) for i in range(min(len(b), len(l)))]
        divergences[sentence_id] = {"first_any_code_divergence": next((i for d, i in distances if d > 0), None), "first_strong_code_divergence_ge_4": next((i for d, i in distances if d >= 4), None), "max_code_distance": max((d for d, _ in distances), default=None)}
    (OUTPUT_DIR / "summary.json").write_text(json.dumps({"config": {"speaker": SPEAKER, "reference": ref_path.name, "temperature": TEMPERATURE, "top_k": TOP_K, "top_p": TOP_P, "repetition_penalty": REPETITION_PENALTY, "seed": SEED, "rms_threshold": RMS_THRESHOLD}, "per_case": {f"{k[0]}_{k[1]}": v for k, v in summary.items()}, "divergence": divergences}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("## Divergence")
    for sentence, result in divergences.items():
        print(sentence, result)
    print(f"output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
