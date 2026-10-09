#!/usr/bin/env python3
"""Generate with the packed regional embedding; no reference WAV is accepted."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from run_card import choose_asr_device, detect_card_profile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE_ROOT = ROOT / "source_code/audio_model"
WORK = HERE / "output/nghean_region_no_anchor"
FALLBACK_RUN = "nghean_region_no_anchor_v3_perclip_medoid_90"
LATEST_RUN = WORK / "latest_run.json"

def latest_merged():
    if LATEST_RUN.is_file():
        try:
            path = Path(json.loads(LATEST_RUN.read_text(encoding="utf-8"))["merged"])
            if (path / "config.json").is_file():
                return path
        except (KeyError, ValueError, OSError, json.JSONDecodeError):
            pass
    return WORK / "training" / FALLBACK_RUN / "merged"

DEFAULT_MERGED = latest_merged()
DEFAULT_OUTPUT = WORK / "demo_medoid_trimmed"
DEFAULT_VOICE = "Nghe An - regional prototype"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    p.add_argument("--voice", default=DEFAULT_VOICE)
    p.add_argument("--style", choices=("tu_nhien", "tin_tuc", "doc_truyen"),
                   default="tu_nhien",
                   help="style token; verify availability in the merged model")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--text", action="append", required=True,
                   help="text to synthesize; repeat the option for multiple WAVs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.30)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--top-p", type=float, default=0.85)
    p.add_argument("--repetition-penalty", type=float, default=1.35)
    p.add_argument("--max-new-frames", type=int, default=1800)
    p.add_argument("--max-chars", type=int, default=180)
    p.add_argument("--context-multiplier", type=int, choices=(1, 2), default=None,
                   help="auto from detected VRAM unless explicitly set")
    p.add_argument("--minor-pause", type=float, default=0.05)
    p.add_argument("--sentence-pause", type=float, default=0.10)
    p.add_argument("--paragraph-pause", type=float, default=0.16)
    p.add_argument("--quiet-threshold-db", type=float, default=-47.0,
                   help="RMS threshold for trimming sustained low-energy breaths")
    p.add_argument("--quiet-max-sec", type=float, default=0.80,
                   help="compress low-energy internal runs longer than this")
    p.add_argument("--quiet-keep-sec", type=float, default=0.18,
                   help="low-energy audio retained after compression")
    p.add_argument("--quality-retries", type=int, default=3,
                   help="retry chunks with silence, long breath/no-word gaps, repeats, or missing speech")
    p.add_argument("--babble-retries", type=int, default=4,
                   help="upstream VieNeu short-chunk repetition guard attempts")
    p.add_argument("--asr-model", choices=("base", "small"), default="small",
                   help="Whisper model for speech/repetition checks")
    p.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto",
                   help="auto selects a dtype supported natively by the detected GPU")
    p.add_argument("--asr-device", choices=("auto", "cpu", "cuda"), default="auto",
                   help="auto uses CUDA only when enough VRAM remains after TTS loads")
    p.add_argument("--no-asr-guard", action="store_true",
                   help="disable CPU ASR checks")
    return p.parse_args()


def resolve(path: Path) -> Path:
    return path.expanduser() if path.is_absolute() else ROOT / path


def join_chunks(chunks, gaps, sr, pauses):
    from vieneu_utils.core_utils import pause_pad_samples, trim_and_fade
    parts = []
    previous_audio = None
    for i, audio in enumerate(chunks):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not audio.size:
            continue
        if previous_audio is not None:
            pause = pauses.get(gaps[i - 1], pauses["sentence"])
            pad = pause_pad_samples(previous_audio, audio, sr, pause)
            if pad:
                parts.append(np.zeros(pad, dtype=np.float32))
        parts.append(audio)
        previous_audio = audio
    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)



def audio_health(audio, sr):
    """Conservative guard for silent/near-silent chunks; does not edit speech."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size < int(0.35 * sr):
        return {"ok": False, "reason": "under_0.35s", "active_ratio": 0.0, "max_internal_quiet_sec": None}
    if not np.isfinite(audio).all():
        return {"ok": False, "reason": "non_finite_samples", "active_ratio": 0.0, "max_internal_quiet_sec": None}
    hop = max(1, int(0.02 * sr))
    n = audio.size // hop
    env = np.sqrt(np.mean(audio[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    db = 20.0 * np.log10(env + 1e-12)
    active = db > -55.0
    active_ratio = float(active.mean()) if active.size else 0.0
    # Ignore edge fade/trim zones; only flag a long internal near-silent span.
    left, right = int(0.20 / 0.02), max(int(0.20 / 0.02), len(active) - int(0.20 / 0.02))
    inner = active[left:right]
    longest = run = 0
    for is_active in inner:
        run = 0 if is_active else run + 1
        longest = max(longest, run)
    max_quiet = longest * 0.02
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    reason = None
    if peak < 0.0018 or active_ratio < 0.08:
        reason = "mostly_silent"
    elif max_quiet > 1.2:
        reason = "internal_quiet_gap_over_1.2s"
    return {"ok": reason is None, "reason": reason, "active_ratio": round(active_ratio, 4),
            "max_internal_quiet_sec": round(max_quiet, 3), "peak": round(peak, 6)}

def compress_long_quiet_runs(audio, sr: int, threshold_db: float = -45.0,
                             max_run_sec: float = 0.65,
                             keep_sec: float = 0.18,
                             frame_sec: float = 0.01,
                             crossfade_sec: float = 0.015):
    """Shorten long breath/silence-like runs while keeping a small natural pause.

    Uses a short-frame RMS envelope alongside VieNeu's edge trimming and
    pause-padding utilities. Only internal low-energy runs above ``max_run_sec``
    are changed; speech and short
    breaths/pauses remain untouched.
    """
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame = max(1, int(round(frame_sec * sr)))
    n = y.size // frame
    if n < 3 or max_run_sec <= 0 or keep_sec < 0:
        return y, {"runs_shortened": 0, "removed_sec": 0.0,
                   "max_run_before_sec": 0.0, "threshold_dbfs": threshold_db}
    rms = np.sqrt(np.mean(y[:n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    quiet = db < threshold_db

    # Bridge only tiny energy spikes (<= 30 ms) inside a breath; this avoids
    # leaving one-frame clicks between pieces of the same quiet run.
    i = 1
    max_hole = max(1, int(round(0.03 / frame_sec)))
    while i < n - 1:
        if quiet[i]:
            i += 1
            continue
        start = i
        while i < n and not quiet[i]:
            i += 1
        if start > 0 and i < n and quiet[start - 1] and quiet[i] and i - start <= max_hole:
            quiet[start:i] = True

    runs = []
    i = 0
    while i < n:
        if not quiet[i]:
            i += 1
            continue
        start = i
        while i < n and quiet[i]:
            i += 1
        end = i
        # Keep leading/trailing edges to the existing trim_and_fade logic.
        if start == 0 or end == n:
            continue
        duration = (end - start) * frame_sec
        if duration > max_run_sec:
            runs.append((start * frame, min(end * frame, y.size), duration))

    if not runs:
        return y, {"runs_shortened": 0, "removed_sec": 0.0,
                   "max_run_before_sec": 0.0, "threshold_dbfs": threshold_db}

    removed_samples = 0
    # Apply from right to left so earlier sample offsets remain valid.
    for start, end, _duration in reversed(runs):
        keep = min(int(round(keep_sec * sr)), end - start)
        head_keep = keep // 2
        tail_keep = keep - head_keep
        cut_start = start + head_keep
        cut_end = end - tail_keep
        if cut_end <= cut_start:
            continue
        nfade = min(int(round(crossfade_sec * sr)), cut_start, y.size - cut_end)
        if nfade <= 0:
            y = np.concatenate((y[:cut_start], y[cut_end:]))
        else:
            left = y[cut_start - nfade:cut_start]
            right = y[cut_end:cut_end + nfade]
            ramp = np.linspace(0.0, 1.0, nfade, endpoint=True, dtype=np.float32)
            blend = left * (1.0 - ramp) + right * ramp
            y = np.concatenate((y[:cut_start - nfade], blend,
                                y[cut_end + nfade:]))
        removed_samples += cut_end - cut_start

    return y, {"runs_shortened": len(runs),
               "removed_sec": round(removed_samples / sr, 3),
               "max_run_before_sec": round(max(x[2] for x in runs), 3),
               "threshold_dbfs": threshold_db,
               "max_run_after_sec": round(keep_sec, 3)}

def _text_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.findall(r"[a-z0-9]+", text)

def _lcs_coverage(predicted: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    row = [0] * (len(expected) + 1)
    for token in predicted:
        current = [0]
        for j, target in enumerate(expected, 1):
            current.append(row[j - 1] + 1 if token == target else max(row[j], current[-1]))
        row = current
    return row[-1] / len(expected)

def _at_most_one_edit(a: str, b: str) -> bool:
    """Return whether normalized strings differ by at most one character."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    i = j = edits = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            edits += 1
            j += 1
            if edits > 1:
                return False
    return True


def _is_expected_repetition(phrase: tuple[str, ...], exp_counts) -> bool:
    """Ignore an ASR phrase close to a repeated phrase in the source text."""
    if exp_counts[phrase] > 1:
        return True
    phrase_text = " ".join(phrase)
    for candidate, count in exp_counts.items():
        if count > 1 and len(candidate) == len(phrase):
            if _at_most_one_edit(phrase_text, " ".join(candidate)):
                return True
    return False


def _repeated_phrase(predicted: list[str], expected: list[str]) -> str | None:
    """Flag clear loops while ignoring isolated ASR/date-token glitches."""
    exp_counts = Counter(tuple(expected[i:i+n])
                         for n in range(2, 6)
                         for i in range(max(0, len(expected) - n + 1)))
    # Whisper expands written area units (m2 / m²) as “met vuong”. Treat
    # repeated numeric area units in the source as expected phrase occurrences.
    area_units = sum(
        i > 0 and expected[i].lower() in {"m", "m2"}
        and expected[i - 1].isdigit()
        for i in range(len(expected)))
    if area_units > 1:
        exp_counts[("met", "vuong")] = area_units

    # Exact adjacent phrase loops are the strongest signal. Ignore any phrase
    # containing a number: Whisper commonly repeats dates and years in isolation.
    for n in range(2, 6):
        for i in range(len(predicted) - 2*n + 1):
            phrase = tuple(predicted[i:i+n])
            if any(token.isdigit() for token in phrase):
                continue
            # A repeated phrase in the ASR transcript is not proof of a model
            # loop when the source text itself uses that phrase more than once.
            if _is_expected_repetition(phrase, exp_counts):
                continue
            if tuple(predicted[i+n:i+2*n]) != phrase:
                continue
            if not any(tuple(expected[j:j+2*n]) == phrase + phrase
                       for j in range(max(0, len(expected) - 2*n + 1))):
                return " ".join(phrase)

    # For non-adjacent recurrence, require several non-overlapping occurrences;
    # a single repeated bigram is common Vietnamese syntax, not evidence of a loop.
    pred_positions: dict[tuple[str, ...], list[int]] = {}
    for n in range(2, 6):
        for i in range(max(0, len(predicted) - n + 1)):
            phrase = tuple(predicted[i:i+n])
            if not any(token.isdigit() for token in phrase):
                pred_positions.setdefault(phrase, []).append(i)
    for phrase, positions in pred_positions.items():
        if _is_expected_repetition(phrase, exp_counts):
            continue
        required = 4 if len(phrase) == 2 else 3
        nonoverlap = []
        for pos in positions:
            if not nonoverlap or pos >= nonoverlap[-1] + len(phrase):
                nonoverlap.append(pos)
        if len(nonoverlap) >= required:
            return " ".join(phrase)
    return None

def asr_health(audio, sr: int, expected_text: str, model, check_gaps: bool = True) -> dict:
    """Check that a synthesized chunk contains recognizable, non-repeated speech."""
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        sf.write(str(wav_path), np.asarray(audio, dtype=np.float32), sr)
        result = model.transcribe(str(wav_path), language="vi",
                                  fp16=(model.device.type == "cuda"),
                                  temperature=0.0, verbose=False,
                                  condition_on_previous_text=True)
    finally:
        wav_path.unlink(missing_ok=True)
    expected = _text_tokens(expected_text)
    predicted = _text_tokens(result.get("text", ""))
    coverage = _lcs_coverage(predicted, expected)
    repeated = _repeated_phrase(predicted, expected)
    speech_segments = [seg for seg in result.get("segments", [])
                       if _text_tokens(seg.get("text", ""))
                       and float(seg.get("no_speech_prob", 0.0)) < 0.7]
    max_gap = max((float(b["start"]) - float(a["end"])
                   for a, b in zip(speech_segments, speech_segments[1:])), default=0.0)
    reason = None
    if len(predicted) < max(3, int(len(expected) * 0.15)):
        reason = "asr_no_speech_or_unintelligible"
    elif coverage < 0.40:
        reason = "asr_low_text_coverage"
    elif repeated:
        reason = "asr_repeated_phrase"
    elif check_gaps and max_gap > 1.4:
        reason = "asr_long_no_word_gap"
    return {"ok": reason is None, "reason": reason,
            "gap_check_applied": check_gaps,
            "text_coverage": round(coverage, 3),
            "recognized_tokens": len(predicted),
            "expected_tokens": len(expected),
            "max_no_word_gap_sec": round(max_gap, 2) if check_gaps else None,
            "repeated_phrase": repeated,
            "recognized_text": result.get("text", "").strip()}

def main():
    args = parse_args()
    try:
        import torch
    except ImportError:
        torch = None
    card = detect_card_profile(torch)
    if args.context_multiplier is None:
        args.context_multiplier = card['context_multiplier']
    run_dtype = card['dtype'] if args.dtype == "auto" else args.dtype
    print(
        f"GPU profile: {card['name']} | VRAM {card['free_vram_gb']:.1f}/"
        f"{card['total_vram_gb']:.1f} GB free/total | "
        f"compute capability {card['compute_capability']} | dtype={run_dtype} | "
        f"context multiplier={args.context_multiplier}", flush=True,
    )
    if args.max_chars < 40:
        raise ValueError("--max-chars must be at least 40")
    merged, output_dir = resolve(args.merged), resolve(args.output_dir)
    if not (merged / "config.json").is_file():
        raise FileNotFoundError(f"Merged model not found: {merged}")
    voices_path = merged / "voices_v3_turbo.json"
    if not voices_path.is_file():
        raise FileNotFoundError(f"Packed regional voice missing: {voices_path}; run train.py first")
    voice_data = json.loads(voices_path.read_text(encoding="utf-8"))
    preset = voice_data.get("presets", {}).get(args.voice)
    if not preset or preset.get("speaker_emb") is None:
        raise ValueError(f"Packed voice {args.voice!r} not found in {voices_path}")
    if preset.get("codes") is not None:
        raise ValueError("Expected embedding-only voice with codes=null")

    # Keep generation within the model's trained context and size the fused
    # output buffer to the requested frame cap. Long MOSS decoding streams to CPU.
    os.environ["VIENEU_FUSED_MAX_FRAMES"] = str(max(512, args.max_new_frames))
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
    from vieneu import Vieneu
    from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps
    from vieneu_utils.core_utils import edge_silence, trim_and_fade

    random.seed(args.seed)
    np.random.seed(args.seed)
    run_device = "cpu"
    if torch is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            run_device = "cuda"

    output_dir.mkdir(parents=True, exist_ok=True)
    tts = Vieneu(mode="v3turbo", backbone_repo=str(merged), device=run_device,
                 dtype=run_dtype, backend="pytorch",
                 babble_retries=args.babble_retries)
    style_labels = getattr(tts.engine.config, "style_labels", None) or {
        "tu_nhien": 16, "tin_tuc": 17, "doc_truyen": 18,
    }
    if args.style not in style_labels:
        raise ValueError(f"Style {args.style!r} is unavailable; model styles: {style_labels}")
    style_id = int(style_labels[args.style])
    # VieNeu's current inference fixes style to config.default_style_token_id;
    # set that token here so this test runner can select a style without a ref WAV.
    tts.engine.config.default_style_token_id = style_id
    print(f"TTS engine device: {tts.engine.device}; style={args.style} (token={style_id})", flush=True)
    trained_limit = int(tts.engine.config.max_position_embeddings)
    runtime_limit = trained_limit * args.context_multiplier
    tts.engine.config.max_position_embeddings = runtime_limit
    tts.engine.model.config.max_position_embeddings = runtime_limit
    tts.engine.model.semantic_backbone.config.max_position_embeddings = runtime_limit
    print(f"runtime token/cache limit: {trained_limit} -> {runtime_limit}; "
          f"fused frame cap={os.environ['VIENEU_FUSED_MAX_FRAMES']}; "
          f"max chars/chunk={args.max_chars}", flush=True)
    if tts.resolve_voice_name(args.voice) != args.voice:
        raise ValueError(f"Voice preset did not load: {args.voice}")
    asr_model = None
    asr_device = None
    if not args.no_asr_guard:
        import whisper
        asr_device = choose_asr_device(torch, args.asr_device, run_device)
        asr_model = whisper.load_model(
            args.asr_model, device=asr_device,
            download_root=str(Path.home() / ".cache/whisper"))
        print(f"ASR guard: Whisper {args.asr_model} on {asr_device.upper()}", flush=True)
    pauses = {"minor": args.minor_pause, "sentence": args.sentence_pause,
              "para": args.paragraph_pause}
    reports = []
    try:
        for index, text in enumerate(args.text, 1):
            chunks, gaps = normalize_to_chunks_v3_with_gaps(text, max_chars=args.max_chars)
            print(f"Generating {index}/{len(args.text)}: {len(chunks)} chunks; voice={args.voice}; style={args.style}; no reference WAV/codes")
            wav_chunks = []
            chunk_checks = []
            for chunk_no, chunk in enumerate(chunks, 1):
                print(f"  chunk {chunk_no}/{len(chunks)}", flush=True)
                accepted = None
                last_health = None
                for attempt in range(args.quality_retries + 1):
                    raw_audio = tts.infer(
                        chunk, voice=args.voice, use_ref_codes=False,
                        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        max_new_frames=args.max_new_frames, max_chars=args.max_chars,
                        batch_size=1, apply_watermark=False,
                    )
                    raw_audio = np.asarray(raw_audio, dtype=np.float32).reshape(-1)
                    lead, tail = edge_silence(raw_audio, tts.sample_rate, thresh_db=-45.0)
                    candidate = trim_and_fade(raw_audio, tts.sample_rate, thresh_db=-45.0,
                                              keep_s=0.04, fade_s=0.015)
                    health = audio_health(candidate, tts.sample_rate)
                    health["attempt"] = attempt + 1
                    health["trimmed_lead_sec"] = round(lead / tts.sample_rate, 3)
                    health["trimmed_tail_sec"] = round(tail / tts.sample_rate, 3)
                    if health["ok"] and asr_model is not None:
                        check = asr_health(candidate, tts.sample_rate, chunk, asr_model)
                        health["asr"] = check
                        if not check["ok"]:
                            health["ok"] = False
                            health["reason"] = check["reason"]
                    if health["ok"]:
                        accepted = candidate
                        last_health = health
                        break
                    last_health = health
                    check = health.get("asr", {})
                    print(f"    retry {attempt + 1}: {health['reason']} "
                          f"active={health['active_ratio']} max_quiet={health['max_internal_quiet_sec']} "
                          f"text_coverage={check.get('text_coverage', 'n/a')} "
                          f"no_word_gap={check.get('max_no_word_gap_sec', 'n/a')}s "
                          f"repeat={check.get('repeated_phrase')}", flush=True)
                if accepted is None:
                    raise RuntimeError(f"Chunk {chunk_no} failed audio quality guard after "
                                       f"{args.quality_retries + 1} attempts: {last_health}")
                wav_chunks.append(accepted)
                chunk_checks.append(last_health)
                print(f"    accepted attempt={last_health['attempt']} "
                      f"active={last_health['active_ratio']} "
                      f"max_quiet={last_health['max_internal_quiet_sec']}s; "
                      f"trim edge lead={last_health['trimmed_lead_sec']}s "
                      f"tail={last_health['trimmed_tail_sec']}s", flush=True)
            audio = join_chunks(wav_chunks, gaps, tts.sample_rate, pauses)
            audio, final_quiet_fix = compress_long_quiet_runs(
                audio, tts.sample_rate,
                threshold_db=args.quiet_threshold_db,
                max_run_sec=args.quiet_max_sec,
                keep_sec=args.quiet_keep_sec,
            )
            final_health = audio_health(audio, tts.sample_rate)
            final_asr = None
            if asr_model is not None:
                final_asr = asr_health(audio, tts.sample_rate, text, asr_model, check_gaps=False)
                if not final_asr["ok"]:
                    raise RuntimeError(f"Full audio failed quality guard: {final_asr}")
            if not final_health["ok"]:
                raise RuntimeError(f"Full audio failed audio guard: {final_health}")
            print(f"  full WAV check: ASR coverage={None if final_asr is None else final_asr['text_coverage']}; "
                  f"long quiet shortened={final_quiet_fix['removed_sec']}s "
                  f"(max before {final_quiet_fix['max_run_before_sec']}s)", flush=True)
            destination = output_dir / f"region_voice_{index:02d}.wav"
            tts.save(audio, str(destination))
            duration = len(audio) / float(tts.sample_rate)
            print(f"saved {destination} duration={duration:.2f}s", flush=True)
            reports.append({
                "text": text, "wav": str(destination), "duration_sec": round(duration, 4),
                "chunks": chunks, "gaps": gaps,
                "sampling": {"temperature": args.temperature, "top_k": args.top_k,
                             "top_p": args.top_p,
                             "repetition_penalty": args.repetition_penalty,
                             "batch_size": 1},
                "voice": args.voice, "reference_audio": None,
                "reference_codes": None, "pauses_sec": pauses,
                "chunk_health": chunk_checks,
                "full_audio_health": {"audio": final_health, "asr": final_asr,
                                      "long_quiet_compression": final_quiet_fix},
                "audio_cleanup": {"edge_trim_threshold_dbfs": -45.0,
                                  "keep_edge_sec": 0.04, "fade_sec": 0.015,
                                  "long_quiet_compression": {
                                      "threshold_dbfs": args.quiet_threshold_db,
                                      "max_run_sec": args.quiet_max_sec,
                                      "keep_sec": args.quiet_keep_sec}},
            })
    finally:
        close = getattr(tts, "close", None)
        if callable(close):
            close()
    (output_dir / "demo_report.json").write_text(json.dumps({
        "merged_model": str(merged), "voice": args.voice,
        "reference_audio": None, "reference_codes": None,
        "card_profile": card,
        "runtime": {"context_multiplier": args.context_multiplier,
                    "model_dtype": run_dtype,
                    "asr_device": asr_device,
                    "token_cache_limit": runtime_limit,
                    "max_chars_per_chunk": args.max_chars,
                    "max_new_frames": args.max_new_frames,
                    "moss_stream_decode_chunk_sec": 8,
                    "asr_guard": None if args.no_asr_guard else args.asr_model,
                    "quiet_compression": {"threshold_dbfs": args.quiet_threshold_db,
                                          "max_run_sec": args.quiet_max_sec,
                                          "keep_sec": args.quiet_keep_sec},
                    "pauses_sec": pauses},
        "results": reports,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DEMO COMPLETE: {output_dir}")


if __name__ == "__main__":
    main()
