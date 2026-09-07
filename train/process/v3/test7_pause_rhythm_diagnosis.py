"""TEST 7: diagnose Base/sampling/LoRA/free-running pause and rhythm issues.

No training, rank change, loss change, or upstream source modification is done.
The generation loop mirrors the local v3 Turbo PyTorch inference loop so EOS and
generated frame counts can be recorded alongside each waveform.
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
OUTPUT_DIR = TRAIN_DIR / "output" / "test7_pause_rhythm_diagnosis"
BASE_CHECKPOINT = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
MODEL_SUBFOLDER = "update"
MOSS_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
SPEAKERS = ["spk_15_0022", "spk_15_0025"]

# sentence_02 and sentence_03 were among the visibly problematic TEST 6 cases.
SENTENCES = [
    ("sentence_01", "Hôm nay thời tiết khá dễ chịu."),
    ("sentence_02", "Chiều nay chúng ta sẽ gặp nhau ở đâu?"),
    ("sentence_03", "Tôi vừa hoàn thành công việc và đang chuẩn bị về nhà."),
    ("sentence_04", "Cậu đã ăn cơm chưa?"),
    ("sentence_06", "Mọi người đang chờ ở phía trước."),
]

MAX_NEW_FRAMES = 300
SEED = 20260827
SAMPLE_RATE = 48000
RMS_WINDOW_MS = 20
RMS_THRESHOLD = 0.01  # fixed for every output (~ -40 dBFS)
SILENCE_MARKS = (0.300, 0.500)

sys.path.insert(0, str(SOURCE_SRC))
sys.path.insert(0, str(TRAIN_DIR))

from lora_one_sample_ffn_test import inject_ffn_lora  # noqa: E402


def load_engine():
    from vieneu._v3_turbo_engine.inference_v3_turbo import VieNeuTTSv3Turbo

    return VieNeuTTSv3Turbo(
        checkpoint_path=BASE_CHECKPOINT,
        model_subfolder=MODEL_SUBFOLDER,
        moss_tokenizer_path=MOSS_REPO,
        device="auto",
        dtype="auto",
    )


def load_rows() -> dict[str, dict[str, str]]:
    with PREFERRED_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {row.get("speakerID", ""): row for row in rows if row.get("speakerID") in SPEAKERS}
    if set(result) != set(SPEAKERS):
        raise RuntimeError(f"Thiếu validation speaker trong preferred CSV: {set(SPEAKERS) - set(result)}")
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_adapter(engine) -> list[str]:
    config_path = ADAPTER_DIR / "adapter_config.json"
    weights_path = ADAPTER_DIR / "adapter_model.pt"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("best_step", -1)) != 100:
        raise RuntimeError(f"Adapter không phải best step 100: {config.get('best_step')}")
    layer = engine.model.acoustic_decoder.layers[0]
    names = inject_ffn_lora(layer, rank=int(config["rank"]), alpha=float(config["alpha"]), dropout=float(config["dropout"]))
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    expected = {name for name, _ in engine.model.named_parameters() if ".lora_A" in name or ".lora_B" in name}
    if set(state) != expected:
        raise RuntimeError(f"Adapter keys mismatch: missing={expected-set(state)}, unexpected={set(state)-expected}")
    _, unexpected = engine.model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected adapter keys: {unexpected}")
    return names


@torch.no_grad()
def generate_codes_with_stop(engine, text: str, reference_data, *, temperature: float, top_k: int, top_p: float, repetition_penalty: float):
    """Mirror source _generate_codes while exposing the stop reason."""
    from vieneu._v3_turbo_engine.rep_history import DEFAULT_REP_WINDOW, RepetitionHistory

    speaker_emb, ref_codes = reference_data
    phonemes = engine._resolve_phonemes(None, text)
    style_id = engine._resolve_style_id()
    spk_t = engine._resolve_speaker_emb(speaker_emb)
    prompt_2d = engine._build_prompt_2d(phonemes, None, ref_codes, style_id)
    input_2d = prompt_2d.unsqueeze(0).to(engine.device)
    model = engine.model
    prefill_embeds = model._build_inputs_embeds(input_2d, speaker_emb=spk_t)
    prefill_out = model.semantic_backbone(inputs_embeds=prefill_embeds, use_cache=True, return_dict=True)
    past_kv = prefill_out.past_key_values
    h = prefill_out.last_hidden_state[:, -1]
    all_codes = []
    eos_id = engine.config.speech_generation_end_token_id
    sgs_id = engine.config.speech_generation_start_token_id
    hist = RepetitionHistory(engine.config.n_vq, DEFAULT_REP_WINDOW) if repetition_penalty != 1.0 else None
    hit_eos = False
    for _ in range(MAX_NEW_FRAMES):
        frame_codes, last_local_out = model.decode_one_frame(
            h, text_token_id=torch.tensor([sgs_id], device=engine.device),
            temperature=temperature, top_k=top_k, audio_top_p=top_p,
            repetition_penalty=repetition_penalty, history_by_channel=hist,
        )
        all_codes.append(frame_codes.cpu())
        text_logits = model.text_lm_head(last_local_out[0, 0]).float()
        if int(text_logits.argmax().item()) == eos_id:
            hit_eos = True
            break
        slot_row = torch.full((1, 1, engine.config.n_vq + 1), engine.config.audio_pad_token_id, dtype=torch.long, device=engine.device)
        engine._prepare_gen_slot_row(slot_row, frame_codes=frame_codes, sgs_id=sgs_id, audio_pad=engine.config.audio_pad_token_id)
        slot_embed = model._build_inputs_embeds(slot_row, speaker_emb=spk_t)
        step_out = model.semantic_backbone(inputs_embeds=slot_embed, past_key_values=past_kv, use_cache=True, return_dict=True)
        past_kv = step_out.past_key_values
        h = step_out.last_hidden_state[:, 0]
    codes = torch.stack(all_codes) if all_codes else torch.zeros(0, engine.config.n_vq, dtype=torch.long)
    stop_reason = "eos" if hit_eos else "max_new_frames"
    return codes, stop_reason


def waveform_metrics(wav: np.ndarray, codes: torch.Tensor) -> dict[str, float | int | str]:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    win = max(1, int(SAMPLE_RATE * RMS_WINDOW_MS / 1000))
    count = len(wav) // win
    rms = np.sqrt(np.mean(wav[: count * win].reshape(count, win) ** 2, axis=1)) if count else np.zeros(0)
    silent = rms < RMS_THRESHOLD
    active = np.flatnonzero(~silent)
    internal_silent = silent.copy()
    if active.size:
        internal_silent[: active[0]] = False
        internal_silent[active[-1] + 1:] = False
    runs = []
    start = None
    for i, flag in enumerate(internal_silent):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((i - start) * RMS_WINDOW_MS / 1000.0)
            start = None
    if start is not None:
        runs.append((len(internal_silent) - start) * RMS_WINDOW_MS / 1000.0)
    frame_np = codes.numpy()
    repeated = int(np.all(frame_np[1:] == frame_np[:-1], axis=1).sum()) if len(frame_np) > 1 else 0
    return {
        "duration_sec": len(wav) / SAMPLE_RATE,
        "generated_frames": int(len(codes)),
        "internal_silence_count": len(runs),
        "total_silence_sec": float(sum(runs)),
        "longest_silence_sec": float(max(runs, default=0.0)),
        "silence_ratio": float(sum(runs) / max(len(wav) / SAMPLE_RATE, 1e-9)),
        "silence_over_300ms": int(sum(x > SILENCE_MARKS[0] for x in runs)),
        "silence_over_500ms": int(sum(x > SILENCE_MARKS[1] for x in runs)),
        "repeated_frame_ratio": float(repeated / max(len(codes) - 1, 1)),
        "repeated_adjacent_frames": repeated,
    }


def run_config(engine, reference_data, text: str, seed: int, config_name: str):
    if config_name.endswith("_det"):
        params = dict(temperature=0.2, top_k=1, top_p=1.0, repetition_penalty=1.0)
    else:
        params = dict(temperature=0.8, top_k=25, top_p=0.95, repetition_penalty=1.2)
    set_seed(seed)
    codes, stop_reason = generate_codes_with_stop(engine, text, reference_data, **params)
    with torch.inference_mode():
        wav = engine._decode_codes(codes)
    return np.asarray(wav, dtype=np.float32).reshape(-1), codes, stop_reason, params


def main() -> None:
    adapter_config = json.loads((ADAPTER_DIR / "adapter_config.json").read_text(encoding="utf-8"))
    if int(adapter_config.get("best_step", -1)) != 100:
        raise RuntimeError("Không có artifact best step 100 để test.")
    rows = load_rows()
    train_speakers = set(adapter_config.get("train_speakers", []))
    if train_speakers.intersection(SPEAKERS):
        raise RuntimeError("Validation speaker bị overlap với train split.")
    engine = load_engine()
    references = {}
    for speaker in SPEAKERS:
        path = CANDIDATES_DIR / rows[speaker]["downloaded_file"]
        references[speaker] = engine.prepare_reference(str(path), denoise=False, use_ref_codes=True)
    print("## TEST 7 — pause/rhythm diagnosis")
    print(f"speakers: {SPEAKERS}; sentences/speaker: {len(SENTENCES)}; outputs: 40")
    print(f"fixed RMS threshold: {RMS_THRESHOLD} ({RMS_WINDOW_MS} ms windows)")
    print(f"Base: {BASE_CHECKPOINT}/{MODEL_SUBFOLDER}")
    print(f"LoRA: {ADAPTER_DIR}; best_step=100")
    records = []
    for mode in ("base", "lora"):
        adapter_names = []
        if mode == "lora":
            adapter_names = load_adapter(engine)
            print(f"LoRA active: {len(adapter_names)} modules")
        for speaker in SPEAKERS:
            for sentence_id, text in SENTENCES:
                for sampling in ("det", "sampling"):
                    config_name = f"{mode}_{sampling}"
                    seed = SEED + abs(hash((speaker, sentence_id))) % 100000
                    wav, codes, stop_reason, params = run_config(engine, references[speaker], text, seed, config_name)
                    filename = f"{sentence_id}_{config_name}.wav"
                    out_path = OUTPUT_DIR / speaker / filename
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(out_path), wav, SAMPLE_RATE, subtype="PCM_16")
                    metrics = waveform_metrics(wav, codes)
                    records.append({
                        "speakerID": speaker, "sentence_id": sentence_id, "text": text,
                        "mode": mode, "sampling": sampling, "output_file": str(Path(speaker) / filename),
                        "reference_file": rows[speaker]["downloaded_file"], "seed": seed,
                        "stop_reason": stop_reason, "hit_eos": stop_reason == "eos",
                        "hit_max_new_frames": stop_reason == "max_new_frames",
                        **params, **metrics,
                    })
                    print(f"generated {speaker} {sentence_id} {config_name}: {metrics['duration_sec']:.2f}s, frames={metrics['generated_frames']}, stop={stop_reason}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with (OUTPUT_DIR / "metadata.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary = {}
    for group in ("base_det", "base_sampling", "lora_det", "lora_sampling"):
        subset = [r for r in records if f"{r['mode']}_{r['sampling']}" == group]
        summary[group] = {k: float(np.mean([r[k] for r in subset])) for k in ("duration_sec", "generated_frames", "internal_silence_count", "total_silence_sec", "longest_silence_sec", "silence_ratio", "repeated_frame_ratio")} | {
            "eos_count": sum(r["hit_eos"] for r in subset), "max_frame_count": sum(r["hit_max_new_frames"] for r in subset),
            "silence_over_300ms": sum(r["silence_over_300ms"] for r in subset), "silence_over_500ms": sum(r["silence_over_500ms"] for r in subset),
        }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("## Summary")
    for group, values in summary.items():
        print(group, json.dumps(values, ensure_ascii=False))
    print(f"output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
