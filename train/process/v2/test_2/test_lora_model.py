"""Runtime inference: VieNeu base + PEFT adapter + NeuCodec.

No weights are merged.  A failed EOS generation is discarded, retried, then
split recursively so one bad word cannot create a runaway WAV or cancel the
rest of the sentence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

CPU_THREADS = max(1, os.cpu_count() or 1)
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, str(CPU_THREADS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = ROOT / "source_code" / "audio_model"
sys.path.insert(0, str(SOURCE_ROOT / "src"))

BASE_MODEL = "pnnbao-ump/VieNeu-TTS-0.3B"
CODEC = "neuphonic/neucodec"
DEFAULT_ADAPTER = ROOT / "train" / "output" / "nghean_v2_lora30_eos" / "adapter"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"
DEFAULT_METADATA = ROOT / "train" / "metadata_na_candidates.csv"
END_TOKEN = "<|SPEECH_GENERATION_END|>"

TEXTS = [
    "Hôm nay tôi đang thử nghiệm giọng nói tiếng Việt.",
    "Tôi xin chào quý vị và các bạn.",
    "Chương trình hôm nay có nhiều thông tin đáng chú ý.",
    "Nghệ An là vùng đất có văn hóa và giọng nói rất riêng.",
    "Mời các bạn cùng theo dõi phần tin tức tiếp theo.",
    "Công việc này cần được thực hiện cẩn thận và đúng quy trình.",
    "Thời tiết hôm nay khá thuận lợi cho các hoạt động ngoài trời.",
    "Tôi sẽ kiểm tra lại thông tin trước khi đưa ra kết luận.",
    "Mọi người hãy giữ gìn sức khỏe và đi lại an toàn.",
    "Xin cảm ơn quý vị đã lắng nghe chương trình.",
]


def choose_reference() -> tuple[Path, str]:
    candidates: list[tuple[int, float, Path, str]] = []
    if DEFAULT_METADATA.is_file():
        with DEFAULT_METADATA.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("split") not in {"valid", "test"}:
                    continue
                if row.get("speakerID") == "spk_37_0224":
                    continue
                try:
                    duration = float(row.get("duration_sec") or 0)
                except ValueError:
                    continue
                path = ROOT / (row.get("local_path") or "")
                text = (row.get("text") or row.get("transcript") or "").strip()
                if path.is_file() and text:
                    candidates.append((0 if 4 <= duration <= 8 else 1, abs(duration - 6), path, text))
    if not candidates:
        raise FileNotFoundError("Không tìm thấy reference valid/test sạch; hãy truyền --reference và --ref-text.")
    candidates.sort(key=lambda x: (x[0], x[1], str(x[2])))
    _, _, path, text = candidates[0]
    return path, text


def dynamic_budget(text: str) -> int:
    words = max(1, len(text.split()))
    chars = max(10, len(text))
    estimated_seconds = max(1.0, chars / 12.0, words / 2.5)
    return max(120, min(700, int(estimated_seconds * 50 * 2.0)))


def expected_duration(text: str) -> float:
    return max(1.0, len(text) / 12.0, len(text.split()) / 2.5)


def split_fragment(text: str) -> tuple[str, str] | None:
    words = text.strip().split()
    if len(words) <= 1:
        return None
    midpoint = len(words) // 2
    return " ".join(words[:midpoint]).strip(), " ".join(words[midpoint:]).strip()


def repeated_pattern(ids: list[int]) -> bool:
    if len(ids) >= 8 and len(set(ids[-8:])) == 1:
        return True
    if len(ids) >= 12 and ids[-12:-6] == ids[-6:]:
        return True
    return False


def infer_one_attempt(engine, text: str, ref_codes, ref_phonemes, temperature: float, top_k: int, greedy: bool):
    import torch

    from vieneu_utils.phonemize_text import phonemize_with_dict

    phonemes = phonemize_with_dict(text, skip_normalize=True)
    prompt_ids = engine._apply_chat_template(ref_codes, ref_phonemes, phonemes)
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=engine.backbone.device).unsqueeze(0)
    end_id = engine.tokenizer.convert_tokens_to_ids(END_TOKEN)
    budget = dynamic_budget(text)
    with torch.no_grad():
        kwargs = dict(
            max_new_tokens=budget,
            min_new_tokens=40,
            eos_token_id=end_id,
            repetition_penalty=1.2,
            use_cache=True,
        )
        if greedy:
            kwargs.update(do_sample=False)
        else:
            kwargs.update(do_sample=True, temperature=temperature, top_k=top_k)
        generated = engine.backbone.generate(prompt, **kwargs)
    tail = generated[0, prompt.shape[-1]:].detach().cpu().tolist()
    if repeated_pattern(tail):
        return None, len(tail), "stall"
    if end_id not in tail:
        return None, len(tail), "missing_eos"
    eos_at = tail.index(end_id)
    speech_ids = tail[:eos_at]
    if not speech_ids:
        return None, len(tail), "empty"
    output_str = engine.tokenizer.decode(speech_ids, add_special_tokens=False)
    audio = engine._decode(output_str)
    duration = len(audio) / engine.sample_rate
    max_duration = min(12.0, max(4.0, expected_duration(text) * 2.2))
    if duration > max_duration:
        return None, len(speech_ids), "runaway"
    return audio, len(speech_ids), "natural"


def generate_safe(engine, text: str, ref_codes, ref_phonemes, stats: dict[str, Any], depth: int = 0):
    attempts = ((0.35, 25, False), (0.20, 15, False), (0.0, 0, True))
    last_reason = "failed"
    for attempt_index, (temperature, top_k, greedy) in enumerate(attempts):
        audio, token_count, reason = infer_one_attempt(
            engine, text, ref_codes, ref_phonemes, temperature, top_k, greedy
        )
        if audio is not None:
            stats["accepted_durations"].append(len(audio) / engine.sample_rate)
            stats["accepted_tokens"].append(token_count)
            if attempt_index == 0:
                stats["natural_eos_success"] += 1
            else:
                stats["retry_success"] += 1
            return audio
        last_reason = reason
        if reason == "missing_eos":
            stats["missing_eos_count"] += 1
        if reason == "runaway":
            stats["runaway_rejected_count"] += 1

    parts = split_fragment(text)
    if parts and depth < 8:
        stats["split_rescue_count"] += 1
        left = generate_safe(engine, parts[0], ref_codes, ref_phonemes, stats, depth + 1)
        right = generate_safe(engine, parts[1], ref_codes, ref_phonemes, stats, depth + 1)
        if left is not None and right is not None:
            import numpy as np
            from vieneu_utils.core_utils import join_audio_chunks
            return join_audio_chunks([left, right], engine.sample_rate, silence_p=0.08)
        if left is not None:
            return left
        if right is not None:
            return right

    stats["atomic_skip_count"] += 1
    print(f"WARNING: loại mảnh không có EOS ({last_reason}): {text}")
    return None


def main() -> None:
    import numpy as np
    import torch
    from vieneu import Vieneu

    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    p.add_argument("--reference", type=Path)
    p.add_argument("--ref-text")
    p.add_argument("--text", action="append", dest="texts")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--seed", type=int, default=37)
    args = p.parse_args()
    adapter = args.adapter if args.adapter.is_absolute() else ROOT / args.adapter
    if not adapter.is_dir():
        raise FileNotFoundError(f"Chưa có LoRA adapter: {adapter}")
    if args.reference:
        reference = args.reference if args.reference.is_absolute() else ROOT / args.reference
        ref_text = args.ref_text or ""
        if not ref_text:
            raise ValueError("--reference cần đi cùng --ref-text")
    else:
        reference, ref_text = choose_reference()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(max(1, min(4, CPU_THREADS)))
    except RuntimeError:
        pass
    cuda = bool(torch.cuda.is_available())
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    texts = args.texts or TEXTS
    print(f"Base model: {BASE_MODEL}")
    print(f"LoRA adapter: {adapter}")
    print("Merged: False")
    print(f"Device: {'cuda' if cuda else 'cpu'}")
    print(f"Reference: {reference}")

    engine = Vieneu(
        mode="standard",
        backbone_repo=BASE_MODEL,
        backbone_device="cuda" if cuda else "cpu",
        codec_repo=CODEC,
        codec_device="cuda" if cuda else "cpu",
        gguf_filename=None,
    )
    stats: dict[str, Any] = {
        "natural_eos_success": 0,
        "retry_success": 0,
        "split_rescue_count": 0,
        "atomic_skip_count": 0,
        "missing_eos_count": 0,
        "runaway_rejected_count": 0,
        "runaway_saved_count": 0,
        "accepted_durations": [],
        "accepted_tokens": [],
    }
    try:
        engine.load_lora_adapter(str(adapter))
        ref_codes, resolved_ref_text = engine._resolve_ref_voice(None, str(reference), None, ref_text)
        ref_phonemes = engine.get_ref_phonemes(resolved_ref_text)
        from vieneu_utils.core_utils import join_audio_chunks
        for index, text in enumerate(texts, 1):
            chunks = text.split(". ") if len(text) > 256 else [text]
            audios = []
            for chunk in chunks:
                audio = generate_safe(engine, chunk, ref_codes, ref_phonemes, stats)
                if audio is not None:
                    audios.append(audio)
            if not audios:
                print(f"SKIP câu {index}: không có fragment đạt EOS")
                continue
            audio = join_audio_chunks(audios, engine.sample_rate, silence_p=0.08)
            duration = len(audio) / engine.sample_rate
            if duration > 12.0:
                stats["runaway_saved_count"] += 1
                print(f"SKIP WAV {index}: duration bất thường {duration:.2f}s")
                continue
            output = output_dir / f"lora_test_{index:02d}.wav"
            engine.save(audio, str(output))
            print(f"{output} | {duration:.2f}s")
    finally:
        engine.close()

    stats["average_duration"] = (
        sum(stats["accepted_durations"]) / len(stats["accepted_durations"])
        if stats["accepted_durations"] else 0.0
    )
    stats["average_generated_tokens"] = (
        sum(stats["accepted_tokens"]) / len(stats["accepted_tokens"])
        if stats["accepted_tokens"] else 0.0
    )
    stats["reference"] = str(reference)
    stats["adapter"] = str(adapter)
    stats["base_model"] = BASE_MODEL
    stats["merged"] = False
    stats["texts_requested"] = len(texts)
    for key in ("accepted_durations", "accepted_tokens"):
        stats.pop(key, None)
    (output_dir / "eos_inference_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
