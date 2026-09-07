"""Quick inference test for the Nghệ An model trained by the advanced pipeline.

The script resolves paths relative to the repository, so it works on Windows
and Ubuntu without hard-coded drive letters.  It uses the merged model by
default and writes test WAV files beside this script.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = ROOT / "source_code" / "audio_model"
sys.path.insert(0, str(SOURCE_ROOT / "src"))

DEFAULT_MODEL = ROOT / "train" / "output" / "nghean_v2_advanced" / "merged_model"
DEFAULT_METADATA = ROOT / "train" / "metadata_na_candidates.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"

TEST_TEXTS = [
    "Hôm nay tôi đang thử nghiệm giọng nói tiếng Việt.",
    "Tôi xin chào quý vị và các bạn.",
    "Chương trình hôm nay có nhiều thông tin đáng chú ý.",
]

# VieNeu v2 normally stops at SPEECH_GENERATION_END.  A bad/stochastic sample
# may omit that token, so the test runner must have a hard fallback limit.
# 700 speech tokens are roughly 10–12 seconds and prevent a 32-second runaway.
MAX_NEW_TOKENS = 700
MIN_NEW_TOKENS = 50
REPETITION_PENALTY = 1.2


def choose_reference() -> tuple[Path, str]:
    """Choose a clean 4–8 second validation reference, excluding old baseline."""
    rows: list[dict[str, str]] = []
    if DEFAULT_METADATA.is_file():
        with DEFAULT_METADATA.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

    candidates = []
    for row in rows:
        if row.get("split") != "valid":
            continue
        if row.get("speakerID") == "spk_37_0224":
            continue
        try:
            duration = float(row.get("duration_sec", "0"))
        except ValueError:
            continue
        rel = row.get("local_path", "")
        path = ROOT / "train" / rel
        if not path.is_file():
            continue
        candidates.append((0 if 4 <= duration <= 8 else 1, abs(duration - 6), path, row.get("text", "")))

    if not candidates:
        raise FileNotFoundError(
            "Không tìm được reference valid 4–8 giây. Hãy truyền --reference và --ref-text."
        )
    candidates.sort(key=lambda item: (item[0], item[1], str(item[2])))
    _, _, path, text = candidates[0]
    return path, text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--ref-text")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--text", action="append", dest="texts")
    args = parser.parse_args()

    model = args.model if args.model.is_absolute() else ROOT / args.model
    if not model.is_dir():
        raise FileNotFoundError(f"Chưa có merged model: {model}")

    if args.reference:
        reference = args.reference if args.reference.is_absolute() else ROOT / args.reference
        ref_text = args.ref_text or ""
        if not ref_text:
            raise ValueError("Khi dùng --reference phải truyền thêm --ref-text.")
    else:
        reference, ref_text = choose_reference()

    from vieneu import Vieneu

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    texts = args.texts or TEST_TEXTS

    print(f"Model: {model}")
    print(f"Reference: {reference}")
    print(f"Output: {output_dir}")

    engine = Vieneu(
        mode="standard",
        backbone_repo=str(model),
        backbone_device="cuda" if _cuda_available() else "cpu",
        codec_repo="neuphonic/neucodec",
        codec_device="cuda" if _cuda_available() else "cpu",
        gguf_filename=None,
    )
    try:
        for index, text in enumerate(texts, 1):
            audio = infer_with_eos_fallback(engine, text, reference, ref_text)
            output = output_dir / f"merged_test_{index:02d}.wav"
            engine.save(audio, str(output))
            duration = len(audio) / engine.sample_rate
            warning = " RUNAWAY_WARNING" if duration > 15 else ""
            print(f"{output} | {duration:.2f}s{warning}")
    finally:
        engine.close()


def infer_with_eos_fallback(engine, text: str, reference: Path, ref_text: str):
    """Infer without hanging when the model omits its speech EOS token.

    The official ``Vieneu.infer`` path uses ``max_length=2048``.  If EOS is
    missing, that can produce a runaway WAV.  We call the same standard-model
    internals with an explicit ``max_new_tokens`` cap.  EOS is still honored
    when present; it is not required for the call to finish.
    """
    import torch
    from vieneu_utils.phonemize_text import phonemize_with_dict, normalize_to_chunks

    ref_codes, resolved_ref_text = engine._resolve_ref_voice(
        None, str(reference), None, ref_text
    )
    chunks = normalize_to_chunks(text, max_chars=256)
    if len(chunks) != 1:
        # The test sentences are short; retain official chunk handling for a
        # caller-provided long sentence rather than silently dropping text.
        return engine.infer(
            text,
            ref_audio=str(reference),
            ref_text=ref_text,
            max_chars=256,
            temperature=0.35,
            top_k=25,
            apply_watermark=False,
        )

    ref_phonemes = engine.get_ref_phonemes(resolved_ref_text)
    phonemes = phonemize_with_dict(chunks[0], skip_normalize=True)
    prompt_ids = engine._apply_chat_template(ref_codes, ref_phonemes, phonemes)
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=engine.backbone.device).unsqueeze(0)
    speech_end_id = engine.tokenizer.convert_tokens_to_ids(
        "<|SPEECH_GENERATION_END|>"
    )

    with torch.no_grad():
        generated = engine.backbone.generate(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=MIN_NEW_TOKENS,
            eos_token_id=speech_end_id,
            do_sample=True,
            temperature=0.35,
            top_k=25,
            repetition_penalty=REPETITION_PENALTY,
            use_cache=True,
        )

    generated_ids = generated[0, prompt.shape[-1]:].detach().cpu().tolist()
    output_str = engine.tokenizer.decode(generated_ids, add_special_tokens=False)
    return engine._decode(output_str)


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


if __name__ == "__main__":
    main()
