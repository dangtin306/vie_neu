"""TEST 6: blind Base vs best-LoRA inference comparison for VieNeu v3 Turbo.

This is an evaluation artifact only. It does not modify upstream VieNeu source,
merge LoRA into the base checkpoint, or train any parameters.
"""
from __future__ import annotations

import csv
import hashlib
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
OUTPUT_DIR = TRAIN_DIR / "output" / "test6_compare"
BASE_CHECKPOINT = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
MODEL_SUBFOLDER = "update"
MOSS_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
SPEAKERS = ["spk_15_0022", "spk_15_0025"]
SENTENCES = [
    "Hôm nay thời tiết khá dễ chịu.",
    "Chiều nay chúng ta sẽ gặp nhau ở đâu?",
    "Tôi vừa hoàn thành công việc và đang chuẩn bị về nhà.",
    "Cậu đã ăn cơm chưa?",
    "Ngày mai có lẽ trời sẽ mưa.",
    "Mọi người đang chờ ở phía trước.",
    "Thật sự hôm nay vui quá!",
    "Tôi muốn nghe lại đoạn này một lần nữa.",
]
TEMPERATURE = 0.8
TOP_K = 25
TOP_P = 0.95
REPETITION_PENALTY = 1.2
MAX_NEW_FRAMES = 300
SEED = 20260827

sys.path.insert(0, str(SOURCE_SRC))
sys.path.insert(0, str(TRAIN_DIR))

from lora_one_sample_ffn_test import inject_ffn_lora  # noqa: E402
from lora_one_sample_test import LoRALinear  # noqa: E402


def load_rows() -> dict[str, dict[str, str]]:
    with PREFERRED_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    found = {row.get("speakerID", ""): row for row in rows if row.get("speakerID") in SPEAKERS}
    missing = [speaker for speaker in SPEAKERS if speaker not in found]
    if missing:
        raise RuntimeError(f"Không tìm thấy validation speaker trong preferred CSV: {missing}")
    for speaker, row in found.items():
        filename = row.get("downloaded_file", "")
        if not filename or not (CANDIDATES_DIR / filename).is_file():
            raise FileNotFoundError(f"Audio reference không tồn tại cho {speaker}: {filename}")
    return found


def load_engine():
    from vieneu._v3_turbo_engine.inference_v3_turbo import VieNeuTTSv3Turbo

    return VieNeuTTSv3Turbo(
        checkpoint_path=BASE_CHECKPOINT,
        model_subfolder=MODEL_SUBFOLDER,
        moss_tokenizer_path=MOSS_REPO,
        device="auto",
        dtype="auto",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_adapter(engine) -> list[str]:
    config_path = ADAPTER_DIR / "adapter_config.json"
    weights_path = ADAPTER_DIR / "adapter_model.pt"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Thiếu adapter artifact trong {ADAPTER_DIR}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("best_step", -1)) != 100:
        raise RuntimeError(f"Adapter không phải best step 100: best_step={config.get('best_step')}")
    targets = config.get("targets")
    expected = [
        "acoustic_decoder.layers.0.attn.qkv",
        "acoustic_decoder.layers.0.attn.o_proj",
        "acoustic_decoder.layers.0.ff_up",
        "acoustic_decoder.layers.0.ff_gate",
        "acoustic_decoder.layers.0.ff_down",
    ]
    if targets != expected:
        raise RuntimeError(f"LoRA targets không khớp TEST 5: {targets}")
    layer = engine.model.acoustic_decoder.layers[0]
    names = inject_ffn_lora(layer, rank=int(config["rank"]), alpha=float(config["alpha"]), dropout=float(config["dropout"]))
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    expected_keys = {
        name for name, parameter in engine.model.named_parameters()
        if ".lora_A" in name or ".lora_B" in name
    }
    if set(state) != expected_keys:
        raise RuntimeError(
            "Adapter keys không khớp model sau khi inject: "
            f"missing={sorted(expected_keys - set(state))}, "
            f"unexpected={sorted(set(state) - expected_keys)}"
        )
    _, unexpected = engine.model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected adapter parameters: {unexpected}")
    return names


def synthesize(engine, text: str, reference_data, seed: int) -> np.ndarray:
    set_seed(seed)
    speaker_emb, ref_codes = reference_data
    with torch.inference_mode():
        wav = engine.infer(
            text=text,
            speaker_emb=speaker_emb,
            ref_codes=ref_codes,
            use_ref_codes=True,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            max_new_frames=MAX_NEW_FRAMES,
            repetition_penalty=REPETITION_PENALTY,
            frame_cap=False,
        )
    return np.asarray(wav, dtype=np.float32).reshape(-1)


def save_wav(path: Path, wav: np.ndarray, sample_rate: int = 48000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sample_rate, subtype="PCM_16")


def blind_label(speaker: str, sentence_id: str) -> str:
    digest = hashlib.sha256(f"{SEED}:{speaker}:{sentence_id}".encode()).hexdigest()
    return "A" if int(digest[:2], 16) % 2 == 0 else "B"


def main() -> None:
    if not ADAPTER_DIR.is_dir():
        raise RuntimeError(f"Không tìm thấy output TEST 5: {ADAPTER_DIR}")
    rows = load_rows()
    train_speakers = set(json.loads((ADAPTER_DIR / "adapter_config.json").read_text(encoding="utf-8")).get("train_speakers", []))
    overlap = train_speakers.intersection(SPEAKERS)
    if overlap:
        raise RuntimeError(f"Validation speaker bị overlap với train: {sorted(overlap)}")

    engine = load_engine()
    print("## TEST 6 — Base vs LoRA inference")
    print(f"validation speakers: {SPEAKERS}")
    print(f"sentences per speaker: {len(SENTENCES)}")
    print(f"base checkpoint: {BASE_CHECKPOINT}/{MODEL_SUBFOLDER}")
    print(f"adapter path: {ADAPTER_DIR}")
    print("adapter checkpoint step: 100")
    print(f"sampling: temperature={TEMPERATURE}, top_k={TOP_K}, top_p={TOP_P}, repetition_penalty={REPETITION_PENALTY}, max_new_frames={MAX_NEW_FRAMES}, seed={SEED}")
    print(f"device: {engine.device}; dtype: {engine.dtype}; backend: PyTorch")

    # Generate one paired sample before the full set to prove the adapter is active.
    probe_speaker = SPEAKERS[0]
    probe_text = SENTENCES[0]
    reference_data = {}
    for speaker in SPEAKERS:
        ref_path = CANDIDATES_DIR / rows[speaker]["downloaded_file"]
        reference_data[speaker] = engine.prepare_reference(str(ref_path), denoise=False, use_ref_codes=True)
        print(f"reference {speaker}: {ref_path.name}; ref_frames={reference_data[speaker][1].shape[0]}")
    print("\n## Adapter activation probe")
    base_probe = synthesize(engine, probe_text, reference_data[probe_speaker], SEED)
    adapter_names = load_adapter(engine)
    lora_probe = synthesize(engine, probe_text, reference_data[probe_speaker], SEED)
    print(f"loaded adapter path: {ADAPTER_DIR}")
    print(f"LoRA module count: {len(adapter_names)}")
    print(f"adapted modules: {adapter_names}")
    print(f"probe Base samples: {base_probe.size}; LoRA samples: {lora_probe.size}")
    print(f"probe max waveform difference: {float(np.max(np.abs(base_probe[:min(base_probe.size, lora_probe.size)] - lora_probe[:min(base_probe.size, lora_probe.size)]))):.8f}")
    if np.array_equal(base_probe, lora_probe):
        raise RuntimeError("LoRA output giống hệt Base ở probe; adapter có thể bị bypass hoặc load sai.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = []
    for speaker in SPEAKERS:
        ref_file = rows[speaker]["downloaded_file"]
        reference = CANDIDATES_DIR / ref_file
        speaker_dir = OUTPUT_DIR / speaker
        for index, text in enumerate(SENTENCES, start=1):
            sentence_id = f"sentence_{index:02d}"
            seed = SEED + index
            base_wav = synthesize(engine, text, reference_data[speaker], seed)
            lora_wav = synthesize(engine, text, reference_data[speaker], seed)
            base_name = f"{sentence_id}_base.wav"
            lora_name = f"{sentence_id}_lora.wav"
            save_wav(speaker_dir / base_name, base_wav)
            save_wav(speaker_dir / lora_name, lora_wav)
            label_a = base_name if blind_label(speaker, sentence_id) == "A" else lora_name
            label_b = lora_name if label_a == base_name else base_name
            metadata.append({
                "speakerID": speaker, "sentence_id": sentence_id, "text": text,
                "reference_file": ref_file, "base_file": str(Path(speaker) / base_name),
                "lora_file": str(Path(speaker) / lora_name), "blind_A": label_a,
                "blind_B": label_b, "seed": seed, "sample_rate": 48000,
                "temperature": TEMPERATURE, "top_k": TOP_K, "top_p": TOP_P,
                "repetition_penalty": REPETITION_PENALTY, "max_new_frames": MAX_NEW_FRAMES,
                "adapter_path": str(ADAPTER_DIR), "adapter_step": 100,
                "base_duration_sec": base_wav.size / 48000.0,
                "lora_duration_sec": lora_wav.size / 48000.0,
            })
            print(f"generated {speaker} {sentence_id}: base={base_wav.size/48000:.2f}s lora={lora_wav.size/48000:.2f}s")

    fields = list(metadata[0])
    with (OUTPUT_DIR / "metadata.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metadata)
    print("\n## Output")
    print(f"folder: {OUTPUT_DIR}")
    print(f"Base WAV: {len(metadata)}")
    print(f"LoRA WAV: {len(metadata)}")
    print("No TEST 7/ASR/F0/subjective scores were fabricated; listen and score the WAV pairs manually.")


if __name__ == "__main__":
    main()
