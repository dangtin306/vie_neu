"""Runtime Base + PEFT LoRA + NeuCodec inference with EOS hard safety.

This tester NEVER accepts a fragment that failed to naturally generate
<|SPEECH_GENERATION_END|>. It retries twice, then recursively splits the text.
Only at an atomic fragment does it try alternate/greedy decoding; if that still
fails, it skips only that fragment and continues.

No model merge is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

CPU_THREADS = max(1, os.cpu_count() or 1)
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, str(CPU_THREADS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = ROOT / "source_code" / "audio_model"
sys.path.insert(0, str(SOURCE_ROOT / "src"))
sys.path.insert(0, str(SOURCE_ROOT))

BASE_MODEL = "pnnbao-ump/VieNeu-TTS-0.3B"
CODEC_MODEL = "neuphonic/neucodec"
DEFAULT_ADAPTER = (
    ROOT
    / "train"
    / "output"
    / "nghean_v2_lora30_eos"
    / "adapter"
)
DEFAULT_METADATA = ROOT / "train" / "metadata_na_candidates.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"

SPEECH_END_TOKEN = "<|SPEECH_GENERATION_END|>"
SPEECH_TOKEN_RATE = 50.0

# The normal/low-temp pair is used on every fragment. If both fail, split NOW.
PRIMARY_PROFILES = (
    ("normal", True, 0.35, 25, 1.12),
    ("low_temp", True, 0.20, 15, 1.15),
)

# Only atomic fragments get these extra attempts before being skipped.
ATOMIC_PROFILES = (
    ("alternate", True, 0.48, 35, 1.08),
    ("greedy", False, None, None, 1.10),
)

DEFAULT_TEST_TEXTS = [
    "Hôm nay tôi đang thử nghiệm giọng nói tiếng Việt.",
    "Tôi xin chào quý vị và các bạn.",
    "Chương trình hôm nay có nhiều thông tin đáng chú ý.",
    "Thời tiết hôm nay khá dễ chịu và có nhiều nắng.",
    "Buổi sáng mọi người thường bắt đầu công việc từ khá sớm.",
    "Tôi muốn kiểm tra cách đọc một câu ngắn và rõ ràng.",
    "Giọng nói cần tự nhiên, dễ nghe và không bị lặp âm.",
    "Thông tin mới sẽ được cập nhật trong chương trình tiếp theo.",
    "Mời quý vị tiếp tục theo dõi những nội dung sau đây.",
    "Hôm nay chúng ta sẽ trao đổi về một chủ đề rất gần gũi.",
    "Các bạn có thể nghe thử để đánh giá chất lượng giọng nói.",
    "Mục tiêu của lần thử này là giữ cách phát âm thật ổn định.",
    "Một câu dài hơn giúp kiểm tra khả năng kết thúc đúng thời điểm.",
    "Xin cảm ơn quý vị đã quan tâm và theo dõi chương trình.",
    "Chúng tôi sẽ quay trở lại với những thông tin mới trong ít phút nữa.",
]


@dataclass
class DecodeProfile:
    name: str
    do_sample: bool
    temperature: float | None
    top_k: int | None
    repetition_penalty: float


@dataclass
class AttemptResult:
    ok: bool
    reason: str
    profile: str
    generated_tokens: int
    token_budget: int
    eos_found: bool
    duration_sec: float | None
    expected_duration_sec: float
    max_allowed_duration_sec: float
    audio: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test VieNeu Base + runtime PEFT LoRA with EOS rescue."
    )
    p.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    p.add_argument("--reference", type=Path)
    p.add_argument("--ref-text")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--text", action="append", dest="texts")
    p.add_argument("--seed", type=int, default=37)
    p.add_argument(
        "--max-split-depth",
        type=int,
        default=6,
    )
    p.add_argument(
        "--token-headroom",
        type=float,
        default=1.75,
        help="Multiplier applied to expected speech token count.",
    )
    p.add_argument(
        "--duration-headroom",
        type=float,
        default=1.75,
        help="Multiplier applied to expected speech duration.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of skipping an unrecoverable atomic fragment.",
    )
    p.add_argument(
        "--compare-base",
        action="store_true",
        help="Also run the same tests on the untouched base model.",
    )
    return p.parse_args()


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_audio_path(raw: str) -> Path | None:
    raw = (raw or "").strip()
    if not raw:
        return None

    p = Path(raw)
    candidates = [p]

    if not p.is_absolute():
        candidates.extend(
            [
                ROOT / raw,
                ROOT / "train" / raw,
            ]
        )

    normalized = raw.replace("\\", "/")
    lower = normalized.lower()

    marker = "/train/"
    idx = lower.find(marker)
    if idx >= 0:
        candidates.append(
            ROOT
            / "train"
            / normalized[idx + len(marker):]
        )

    if lower.startswith("train/"):
        candidates.append(ROOT / normalized)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)

        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            pass

    return None


def choose_reference() -> tuple[Path, str]:
    if not DEFAULT_METADATA.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy metadata: {DEFAULT_METADATA}"
        )

    with DEFAULT_METADATA.open(
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    candidates = []

    for row in rows:
        if (row.get("split") or "").strip().lower() not in {
            "valid",
            "test",
        }:
            continue

        # Exclude the previously problematic baseline reference.
        if (row.get("speakerID") or "").strip() == "spk_37_0224":
            continue

        try:
            duration = float(
                row.get("duration_sec")
                or row.get("duration")
                or row.get("duration_metadata")
                or 0
            )
        except (TypeError, ValueError):
            continue

        path = resolve_audio_path(
            row.get("local_path")
            or row.get("downloaded_file")
            or ""
        )
        if path is None:
            continue

        text = (
            row.get("transcript")
            or row.get("text")
            or row.get("transcription")
            or ""
        ).strip()

        if not text:
            continue

        candidates.append(
            (
                0 if 4.0 <= duration <= 8.0 else 1,
                abs(duration - 6.0),
                path,
                text,
            )
        )

    if not candidates:
        raise FileNotFoundError(
            "Không tìm được reference valid/test; "
            "hãy truyền --reference và --ref-text."
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
            str(item[2]),
        )
    )

    _, _, path, text = candidates[0]
    return path, text


def expected_duration(text: str) -> float:
    """Conservative Vietnamese speech-duration estimate for safety gating."""
    clean = re.sub(r"\s+", " ", text.strip())
    words = max(1, len(clean.split()))
    chars = max(1, len(clean))

    # About 2.7-3.2 words/s in ordinary TTS, with a small char correction.
    by_words = words / 2.8
    by_chars = chars / 15.0
    return max(0.65, max(by_words, by_chars))


def token_budget(
    text: str,
    prompt_len: int,
    max_context: int,
    token_headroom: float,
) -> tuple[int, float]:
    expected_sec = expected_duration(text)
    expected_tokens = expected_sec * SPEECH_TOKEN_RATE

    budget = int(
        expected_tokens * max(1.1, token_headroom)
        + 30
    )

    # Short fragments should never inherit a ~500-token budget.
    min_budget = 55
    max_budget = 420
    budget = max(min_budget, min(max_budget, budget))

    remaining_context = max(
        1,
        int(max_context) - int(prompt_len) - 2,
    )
    budget = max(1, min(budget, remaining_context))

    return budget, expected_sec


def min_new_tokens_for(text: str, budget: int) -> int:
    words = max(1, len(text.split()))
    value = 8 + words * 3
    return max(
        8,
        min(42, value, max(8, budget // 3)),
    )


class StallDetector:
    """Early-stop obvious pathological token loops."""

    def __init__(self, prompt_len: int):
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        seq = (
            input_ids[0, self.prompt_len:]
            .detach()
            .cpu()
            .tolist()
        )
        n = len(seq)

        if n < 60:
            return False

        # Same token repeated 10 times.
        if len(set(seq[-10:])) == 1:
            return True

        # Same 5-token pattern repeated 5 times.
        pattern_len = 5
        repeats = 5
        if n >= pattern_len * repeats:
            tail = seq[-pattern_len:]
            ok = True
            for i in range(repeats):
                end = n - pattern_len * i
                start = end - pattern_len
                if seq[start:end] != tail:
                    ok = False
                    break
            if ok:
                return True

        return False


def make_stopping_criteria(prompt_len: int):
    from transformers import (
        StoppingCriteria,
        StoppingCriteriaList,
    )

    detector = StallDetector(prompt_len)

    class HFStoppingCriteria(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return detector(
                input_ids,
                scores,
                **kwargs,
            )

    return StoppingCriteriaList(
        [HFStoppingCriteria()]
    )


def normalize_profile(raw) -> DecodeProfile:
    return DecodeProfile(*raw)


def generate_one_attempt(
    engine: Any,
    text: str,
    ref_codes: Any,
    ref_phonemes: str,
    profile: DecodeProfile,
    seed: int,
    token_headroom: float,
    duration_headroom: float,
) -> AttemptResult:
    import torch
    from vieneu_utils.phonemize_text import phonemize_with_dict

    seed_everything(seed)

    phonemes = phonemize_with_dict(
        text,
        skip_normalize=True,
    )

    prompt_ids = engine._apply_chat_template(
        ref_codes,
        ref_phonemes,
        phonemes,
    )

    prompt = torch.tensor(
        prompt_ids,
        dtype=torch.long,
        device=engine.backbone.device,
    ).unsqueeze(0)

    speech_end_id = engine.tokenizer.convert_tokens_to_ids(
        SPEECH_END_TOKEN
    )

    if speech_end_id is None or int(speech_end_id) < 0:
        raise RuntimeError(
            f"Tokenizer không có {SPEECH_END_TOKEN}."
        )

    budget, expected_sec = token_budget(
        text,
        prompt.shape[-1],
        engine.max_context,
        token_headroom,
    )

    max_allowed_sec = (
        expected_sec * max(1.1, duration_headroom)
        + 0.75
    )

    kwargs = dict(
        max_new_tokens=budget,
        min_new_tokens=min_new_tokens_for(text, budget),
        eos_token_id=int(speech_end_id),
        do_sample=profile.do_sample,
        repetition_penalty=profile.repetition_penalty,
        use_cache=True,
        stopping_criteria=make_stopping_criteria(
            prompt.shape[-1]
        ),
        renormalize_logits=True,
        remove_invalid_values=True,
    )

    if profile.do_sample:
        kwargs["temperature"] = profile.temperature
        kwargs["top_k"] = profile.top_k

    try:
        with torch.inference_mode():
            generated = engine.backbone.generate(
                prompt,
                **kwargs,
            )
    except Exception as exc:
        return AttemptResult(
            ok=False,
            reason=f"generate_error:{type(exc).__name__}:{exc}",
            profile=profile.name,
            generated_tokens=0,
            token_budget=budget,
            eos_found=False,
            duration_sec=None,
            expected_duration_sec=expected_sec,
            max_allowed_duration_sec=max_allowed_sec,
        )

    new_ids = (
        generated[0, prompt.shape[-1]:]
        .detach()
        .cpu()
        .tolist()
    )

    if int(speech_end_id) not in new_ids:
        return AttemptResult(
            ok=False,
            reason="missing_eos_or_stall",
            profile=profile.name,
            generated_tokens=len(new_ids),
            token_budget=budget,
            eos_found=False,
            duration_sec=None,
            expected_duration_sec=expected_sec,
            max_allowed_duration_sec=max_allowed_sec,
        )

    eos_position = new_ids.index(int(speech_end_id))
    speech_ids = new_ids[:eos_position]

    if not speech_ids:
        return AttemptResult(
            ok=False,
            reason="empty_before_eos",
            profile=profile.name,
            generated_tokens=len(new_ids),
            token_budget=budget,
            eos_found=True,
            duration_sec=None,
            expected_duration_sec=expected_sec,
            max_allowed_duration_sec=max_allowed_sec,
        )

    # Reject a suspiciously late EOS before codec decode when token count alone
    # is already wildly larger than the expected speech duration.
    max_reasonable_speech_tokens = int(
        max_allowed_sec * SPEECH_TOKEN_RATE
    )
    if len(speech_ids) > max_reasonable_speech_tokens:
        return AttemptResult(
            ok=False,
            reason=(
                "late_eos_by_tokens:"
                f"{len(speech_ids)}>{max_reasonable_speech_tokens}"
            ),
            profile=profile.name,
            generated_tokens=len(new_ids),
            token_budget=budget,
            eos_found=True,
            duration_sec=None,
            expected_duration_sec=expected_sec,
            max_allowed_duration_sec=max_allowed_sec,
        )

    try:
        output_string = engine.tokenizer.decode(
            speech_ids,
            add_special_tokens=False,
        )
        audio = engine._decode(output_string)
    except Exception as exc:
        return AttemptResult(
            ok=False,
            reason=f"decode_error:{type(exc).__name__}:{exc}",
            profile=profile.name,
            generated_tokens=len(new_ids),
            token_budget=budget,
            eos_found=True,
            duration_sec=None,
            expected_duration_sec=expected_sec,
            max_allowed_duration_sec=max_allowed_sec,
        )

    if audio is None or len(audio) == 0:
        return AttemptResult(
            ok=False,
            reason="empty_audio",
            profile=profile.name,
            generated_tokens=len(new_ids),
            token_budget=budget,
            eos_found=True,
            duration_sec=0.0,
            expected_duration_sec=expected_sec,
            max_allowed_duration_sec=max_allowed_sec,
        )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )
    duration_sec = len(audio) / engine.sample_rate

    if duration_sec > max_allowed_sec:
        return AttemptResult(
            ok=False,
            reason=(
                "late_eos_by_audio:"
                f"{duration_sec:.2f}s>{max_allowed_sec:.2f}s"
            ),
            profile=profile.name,
            generated_tokens=len(new_ids),
            token_budget=budget,
            eos_found=True,
            duration_sec=duration_sec,
            expected_duration_sec=expected_sec,
            max_allowed_duration_sec=max_allowed_sec,
        )

    return AttemptResult(
        ok=True,
        reason="ok",
        profile=profile.name,
        generated_tokens=len(new_ids),
        token_budget=budget,
        eos_found=True,
        duration_sec=duration_sec,
        expected_duration_sec=expected_sec,
        max_allowed_duration_sec=max_allowed_sec,
        audio=audio,
    )


def split_fragment(text: str) -> tuple[str, str] | None:
    text = re.sub(r"\s+", " ", text.strip())
    words = text.split()

    if len(words) <= 1:
        return None

    midpoint = len(text) // 2

    punctuation_positions = [
        match.end()
        for match in re.finditer(
            r"[,;:!?\.]\s+",
            text,
        )
        if 0 < match.end() < len(text)
    ]

    if punctuation_positions:
        cut = min(
            punctuation_positions,
            key=lambda value: abs(value - midpoint),
        )
        left = text[:cut].strip()
        right = text[cut:].strip()
        if left and right:
            return left, right

    middle_word = len(words) // 2
    left = " ".join(words[:middle_word]).strip()
    right = " ".join(words[middle_word:]).strip()

    if left and right:
        return left, right

    return None


def concatenate_audio(
    pieces: list[np.ndarray],
    sample_rate: int,
    silence_sec: float = 0.035,
) -> np.ndarray:
    if not pieces:
        return np.array([], dtype=np.float32)

    if len(pieces) == 1:
        return pieces[0]

    silence = np.zeros(
        max(1, int(sample_rate * silence_sec)),
        dtype=np.float32,
    )

    output = []
    for index, piece in enumerate(pieces):
        if index:
            output.append(silence)
        output.append(piece)

    return np.concatenate(output)


def append_attempt_report(
    report: list[dict[str, Any]],
    text: str,
    depth: int,
    attempt_number: int,
    result: AttemptResult,
    stage: str,
) -> None:
    row = {
        "text": text,
        "depth": depth,
        "attempt": attempt_number,
        "stage": stage,
    }
    row.update(
        {
            key: value
            for key, value in asdict(result).items()
            if key != "audio"
        }
    )
    report.append(row)


def synthesize_fragment(
    engine: Any,
    text: str,
    ref_codes: Any,
    ref_phonemes: str,
    seed: int,
    depth: int,
    max_depth: int,
    token_headroom: float,
    duration_headroom: float,
    strict: bool,
    report: list[dict[str, Any]],
    counters: dict[str, int],
) -> list[np.ndarray]:
    # Phase 1: exactly two attempts on a normal fragment.
    for attempt_index, raw_profile in enumerate(
        PRIMARY_PROFILES,
        start=1,
    ):
        profile = normalize_profile(raw_profile)
        result = generate_one_attempt(
            engine,
            text,
            ref_codes,
            ref_phonemes,
            profile,
            seed + attempt_index * 1009 + depth * 7919,
            token_headroom,
            duration_headroom,
        )

        append_attempt_report(
            report,
            text,
            depth,
            attempt_index,
            result,
            "primary",
        )

        if not result.eos_found:
            counters["missing_eos_count"] += 1

        if result.ok and result.audio is not None:
            if depth == 0 and attempt_index == 1:
                counters["natural_eos_success"] += 1
            else:
                counters["retry_success"] += 1

            print(
                f"  ✅ EOS [{profile.name}] depth={depth} "
                f"tokens={result.generated_tokens}/{result.token_budget} "
                f"duration={result.duration_sec:.2f}s | {text!r}",
                flush=True,
            )
            return [result.audio]

        print(
            f"  ⚠️ [{profile.name}] depth={depth} "
            f"{result.reason} "
            f"tokens={result.generated_tokens}/{result.token_budget} "
            f"| {text!r}",
            flush=True,
        )

    # Phase 2: after two failures, split immediately if possible.
    if depth < max_depth:
        split = split_fragment(text)
        if split is not None:
            left, right = split
            counters["split_rescue_count"] += 1

            print(
                f"  🔀 split NGAY depth={depth}: "
                f"{left!r} | {right!r}",
                flush=True,
            )

            left_audio = synthesize_fragment(
                engine,
                left,
                ref_codes,
                ref_phonemes,
                seed + 17,
                depth + 1,
                max_depth,
                token_headroom,
                duration_headroom,
                strict,
                report,
                counters,
            )
            right_audio = synthesize_fragment(
                engine,
                right,
                ref_codes,
                ref_phonemes,
                seed + 31,
                depth + 1,
                max_depth,
                token_headroom,
                duration_headroom,
                strict,
                report,
                counters,
            )
            return left_audio + right_audio

    # Phase 3: atomic fragment only. Two last-resort profiles.
    for offset, raw_profile in enumerate(
        ATOMIC_PROFILES,
        start=1,
    ):
        profile = normalize_profile(raw_profile)
        attempt_number = 2 + offset

        result = generate_one_attempt(
            engine,
            text,
            ref_codes,
            ref_phonemes,
            profile,
            seed + 50_000 + offset * 1237 + depth * 7919,
            token_headroom,
            duration_headroom,
        )

        append_attempt_report(
            report,
            text,
            depth,
            attempt_number,
            result,
            "atomic_last_resort",
        )

        if not result.eos_found:
            counters["missing_eos_count"] += 1

        if result.ok and result.audio is not None:
            counters["retry_success"] += 1

            print(
                f"  ✅ atomic EOS [{profile.name}] depth={depth} "
                f"tokens={result.generated_tokens}/{result.token_budget} "
                f"| {text!r}",
                flush=True,
            )
            return [result.audio]

        print(
            f"  ⚠️ atomic [{profile.name}] depth={depth} "
            f"{result.reason} "
            f"| {text!r}",
            flush=True,
        )

    if strict:
        raise RuntimeError(
            f"Không cứu được atomic fragment: {text!r}"
        )

    counters["atomic_skip_count"] += 1
    report.append(
        {
            "text": text,
            "depth": depth,
            "attempt": "final",
            "stage": "skip",
            "ok": False,
            "reason": "skipped_unrecoverable_atomic_fragment",
        }
    )

    print(
        f"  ⏭️ BỎ fragment lỗi và đọc tiếp: {text!r}",
        flush=True,
    )
    return []


def prepare_reference(
    engine: Any,
    reference: Path,
    ref_text: str,
):
    ref_codes, resolved_ref_text = engine._resolve_ref_voice(
        None,
        str(reference),
        None,
        ref_text,
    )
    ref_phonemes = engine.get_ref_phonemes(
        resolved_ref_text
    )
    return ref_codes, ref_phonemes


def infer_text(
    engine: Any,
    text: str,
    ref_codes: Any,
    ref_phonemes: str,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    from vieneu_utils.phonemize_text import normalize_to_chunks

    chunks = normalize_to_chunks(
        text,
        max_chars=180,
    )

    attempts: list[dict[str, Any]] = []
    counters = {
        "natural_eos_success": 0,
        "retry_success": 0,
        "split_rescue_count": 0,
        "atomic_skip_count": 0,
        "missing_eos_count": 0,
        "runaway_saved_count": 0,
    }
    audio_pieces: list[np.ndarray] = []

    for chunk_index, chunk in enumerate(chunks, start=1):
        print(
            f"\n🗣️ chunk {chunk_index}/{len(chunks)}: {chunk}",
            flush=True,
        )

        audio_pieces.extend(
            synthesize_fragment(
                engine,
                chunk,
                ref_codes,
                ref_phonemes,
                seed + chunk_index * 100_003,
                0,
                max(0, args.max_split_depth),
                args.token_headroom,
                args.duration_headroom,
                args.strict,
                attempts,
                counters,
            )
        )

    audio = concatenate_audio(
        audio_pieces,
        engine.sample_rate,
    )
    return audio, attempts, counters


def load_engine(
    adapter: Path | None,
    device: str,
):
    from peft import PeftModel
    from vieneu import Vieneu

    engine = Vieneu(
        mode="standard",
        backbone_repo=BASE_MODEL,
        backbone_device=device,
        codec_repo=CODEC_MODEL,
        codec_device=device,
        gguf_filename=None,
    )

    if adapter is not None:
        if not adapter.is_dir():
            engine.close()
            raise FileNotFoundError(
                f"Adapter không tồn tại: {adapter}"
            )

        # Runtime PEFT injection. The original backbone remains the base model;
        # no merged checkpoint is created.
        engine.backbone = PeftModel.from_pretrained(
            engine.backbone,
            str(adapter),
            is_trainable=False,
        )
        engine.backbone = engine.backbone.to(device)
        engine.backbone.eval()

    return engine


def run_model_tests(
    label: str,
    adapter: Path | None,
    reference: Path,
    ref_text: str,
    texts: list[str],
    output_root: Path,
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    engine = load_engine(
        adapter=adapter,
        device=device,
    )

    output_dir = output_root / label
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregate = {
        "natural_eos_success": 0,
        "retry_success": 0,
        "split_rescue_count": 0,
        "atomic_skip_count": 0,
        "missing_eos_count": 0,
        "runaway_saved_count": 0,
    }

    results = []
    durations = []
    accepted_generated_tokens = []

    print(
        f"\n{'=' * 80}\n"
        f"MODEL LABEL : {label}\n"
        f"Base model  : {BASE_MODEL}\n"
        f"LoRA adapter: {adapter if adapter is not None else 'NONE'}\n"
        f"Merged      : False\n"
        f"Device      : {device}\n"
        f"{'=' * 80}",
        flush=True,
    )

    try:
        ref_codes, ref_phonemes = prepare_reference(
            engine,
            reference,
            ref_text,
        )

        for index, text in enumerate(texts, start=1):
            print(
                f"\n{'-' * 80}\n"
                f"TEST {index:02d}: {text}",
                flush=True,
            )

            audio, attempts, counters = infer_text(
                engine,
                text,
                ref_codes,
                ref_phonemes,
                args,
                args.seed + index * 1_000_003,
            )

            for key in aggregate:
                aggregate[key] += counters[key]

            wav_path = output_dir / f"{label}_{index:02d}.wav"

            saved = False
            duration = 0.0

            if len(audio) > 0:
                engine.save(
                    audio,
                    str(wav_path),
                )
                saved = True
                duration = len(audio) / engine.sample_rate
                durations.append(duration)

            for attempt in attempts:
                if attempt.get("ok") is True:
                    accepted_generated_tokens.append(
                        int(attempt.get("generated_tokens", 0) or 0)
                    )

            results.append(
                {
                    "index": index,
                    "text": text,
                    "wav": str(wav_path) if saved else None,
                    "saved": saved,
                    "duration_sec": round(duration, 3),
                    "counters": counters,
                    "attempts": attempts,
                }
            )

            print(
                f"✅ result saved={saved} duration={duration:.2f}s "
                f"split={counters['split_rescue_count']} "
                f"skip={counters['atomic_skip_count']} "
                f"missing_eos_attempts={counters['missing_eos_count']}",
                flush=True,
            )

    finally:
        engine.close()

    # Safety invariant: this script never decodes a missing-EOS fragment, and
    # it rejects implausibly late EOS before accepting the fragment.
    aggregate["runaway_saved_count"] = 0

    return {
        "label": label,
        "base_model": BASE_MODEL,
        "adapter": str(adapter) if adapter is not None else None,
        "merged": False,
        "device": device,
        "reference": str(reference),
        "test_count": len(texts),
        **aggregate,
        "average_duration_sec": (
            round(sum(durations) / len(durations), 3)
            if durations
            else 0.0
        ),
        "average_generated_tokens": (
            round(
                sum(accepted_generated_tokens)
                / len(accepted_generated_tokens),
                2,
            )
            if accepted_generated_tokens
            else 0.0
        ),
        "results": results,
    }


def main() -> None:
    args = parse_args()

    import torch

    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(
            max(1, min(4, CPU_THREADS))
        )
    except RuntimeError:
        pass

    adapter = args.adapter.expanduser()
    if not adapter.is_absolute():
        adapter = ROOT / adapter

    if not adapter.is_dir():
        raise FileNotFoundError(
            f"Adapter không tồn tại: {adapter}"
        )

    if args.reference is not None:
        reference = args.reference.expanduser()
        if not reference.is_absolute():
            reference = ROOT / reference

        if not reference.is_file():
            raise FileNotFoundError(
                f"Reference không tồn tại: {reference}"
            )

        ref_text = (args.ref_text or "").strip()
        if not ref_text:
            raise ValueError(
                "Khi truyền --reference phải truyền --ref-text."
            )
    else:
        reference, ref_text = choose_reference()

    output_root = args.output_dir.expanduser()
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    texts = args.texts or DEFAULT_TEST_TEXTS
    device = "cuda" if torch.cuda.is_available() else "cpu"

    reports = []

    if args.compare_base:
        reports.append(
            run_model_tests(
                label="base",
                adapter=None,
                reference=reference,
                ref_text=ref_text,
                texts=texts,
                output_root=output_root,
                args=args,
                device=device,
            )
        )

    reports.append(
        run_model_tests(
            label="lora",
            adapter=adapter,
            reference=reference,
            ref_text=ref_text,
            texts=texts,
            output_root=output_root,
            args=args,
            device=device,
        )
    )

    final_report = {
        "base_model": BASE_MODEL,
        "codec_model": CODEC_MODEL,
        "runtime_adapter": str(adapter),
        "merged": False,
        "device": device,
        "test_text_count": len(texts),
        "token_headroom": args.token_headroom,
        "duration_headroom": args.duration_headroom,
        "max_split_depth": args.max_split_depth,
        "models": reports,
    }

    report_path = output_root / "eos_inference_report.json"
    report_path.write_text(
        json.dumps(
            final_report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # This invariant is intentional and should remain true even when the model
    # itself misses EOS frequently.
    for report in reports:
        if report["runaway_saved_count"] != 0:
            raise RuntimeError(
                "Safety invariant broken: runaway_saved_count != 0"
            )

    print(
        f"\n📄 EOS report: {report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
