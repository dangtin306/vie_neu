#!/usr/bin/env python3
"""Review test_6 transcripts with WhisperX Vietnamese forced alignment.

This creates review artifacts only. It never edits metadata/audio or trains a model.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import os
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = HERE / "output/nghean_region_no_anchor"
DEFAULT_DATASET = WORK / "dataset"
DEFAULT_OUT = WORK / "transcript_alignment"
DEFAULT_HF_HOME = Path("/home/mediatech/dangtin/trainning/application/huggingface")
DEFAULT_ALIGN_MODEL = "dragonSwing/wav2vec2-base-vietnamese"
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metadata", type=Path, default=None,
                        help="default: DATASET_DIR/metadata.csv (file|transcript)")
    parser.add_argument("--audio-dir", type=Path, default=None,
                        help="default: DATASET_DIR/raw_audio")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--align-model", default=DEFAULT_ALIGN_MODEL)
    parser.add_argument("--asr-model", default="small",
                        help="WhisperX ASR model used only as a transcript cross-check")
    parser.add_argument("--skip-asr", action="store_true",
                        help="only run forced alignment; skip ASR transcript comparison")
    parser.add_argument("--min-word-score", type=float, default=None,
                        help="optional score threshold; WhisperX scores are not calibrated word confidence")
    parser.add_argument("--max-asr-wer", type=float, default=0.20,
                        help="flag transcript/ASR disagreement above this WER")
    parser.add_argument("--files", nargs="*", default=None,
                        help="align selected WAV basenames; default: all metadata rows")
    parser.add_argument("--limit", type=int, default=0,
                        help="process only the first N rows (useful for a pilot)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME,
                        help="Hugging Face cache root")
    return parser.parse_args()


def read_metadata(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_no, row in enumerate(csv.reader(stream, delimiter="|"), 1):
            if not row or not any(part.strip() for part in row):
                continue
            if len(row) < 2:
                raise ValueError(f"{path}:{line_no}: expected file_name|transcript")
            filename, text = row[0].strip(), "|".join(row[1:]).strip()
            if Path(filename).name != filename or filename in {".", ".."}:
                raise ValueError(f"{path}:{line_no}: unsafe audio filename {filename!r}")
            if not text:
                raise ValueError(f"{path}:{line_no}: empty transcript for {filename}")
            rows.append((filename, text))
    if not rows:
        raise ValueError(f"No transcript rows found in {path}")
    return rows


def words(text: str):
    normalized = unicodedata.normalize("NFC", text).casefold()
    return WORD_RE.findall(normalized)

def fold_vietnamese_word(word: str):
    decomposed = unicodedata.normalize("NFD", word.casefold())
    bare = "".join(char for char in decomposed if not unicodedata.combining(char))
    return bare.replace("đ", "d")


def word_error_rate(reference: list[str], hypothesis: list[str]) -> float | None:
    if not reference:
        return None
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, 1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ref_word != hyp_word),
            ))
        previous = current
    return previous[-1] / len(reference)


def transcript_diffs(reference: list[str], hypothesis: list[str]):
    result = []
    ref_fold = [fold_vietnamese_word(word) for word in reference]
    hyp_fold = [fold_vietnamese_word(word) for word in hypothesis]
    matcher = difflib.SequenceMatcher(a=ref_fold, b=hyp_fold, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            result.append({"kind": tag, "transcript": reference[i1:i2],
                           "asr": hypothesis[j1:j2]})
    return result


def finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def main():
    args = parse_args()
    metadata = args.metadata or (args.dataset_dir / "metadata.csv")
    audio_dir = args.audio_dir or (args.dataset_dir / "raw_audio")
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    if not audio_dir.is_dir():
        raise FileNotFoundError(audio_dir)
    if args.limit < 0 or args.batch_size < 1:
        raise ValueError("--limit must be >= 0 and --batch-size must be >= 1")
    if (args.min_word_score is not None and not 0 <= args.min_word_score <= 1) or not 0 <= args.max_asr_wer <= 1:
        raise ValueError("score/wer thresholds must be between 0 and 1")

    args.hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(args.hf_home / "hub"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(args.hf_home / "hub"))

    import torch
    import whisperx

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    compute_type = "float16" if device == "cuda" else "int8"

    rows = read_metadata(metadata)
    if args.files is not None:
        selected = set(args.files)
        rows = [row for row in rows if row[0] in selected]
        missing = selected - {name for name, _ in rows}
        if missing:
            raise FileNotFoundError(f"Requested names absent from metadata: {sorted(missing)}")
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("No clips selected")

    print(f"WhisperX {__import__('importlib.metadata').metadata.version('whisperx')} | device={device} | compute={compute_type}", flush=True)
    print(f"dataset rows={len(rows)} | metadata={metadata} | output={args.out_dir}", flush=True)
    print(f"loading Vietnamese aligner: {args.align_model}", flush=True)
    align_model, align_metadata = whisperx.load_align_model(
        language_code="vi", device=device, model_name=args.align_model,
        model_dir=str(args.hf_home / "hub"),
    )

    asr_model = None
    if not args.skip_asr:
        print(f"loading ASR cross-check: {args.asr_model}", flush=True)
        asr_model = whisperx.load_model(
            args.asr_model, device, compute_type=compute_type,
            language="vi", task="transcribe", vad_method="silero",
            download_root=str(args.hf_home / "whisperx"),
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.out_dir / "aligned_words.jsonl"
    output_words = args.out_dir / "aligned_words.csv"
    output_review = args.out_dir / "review.csv"
    output_summary = args.out_dir / "summary.json"
    result_rows, word_rows, review_rows = [], [], []

    for index, (filename, transcript) in enumerate(rows, 1):
        wav_path = audio_dir / filename
        if not wav_path.is_file():
            raise FileNotFoundError(wav_path)
        audio = whisperx.load_audio(str(wav_path))
        duration = len(audio) / 16000.0
        if duration <= 0:
            raise ValueError(f"Empty audio: {wav_path}")

        asr_text = None
        asr_wer = None
        asr_wer_strict = None
        asr_segments = []
        asr_start = asr_end = None
        differences = []
        if asr_model is not None:
            asr_result = asr_model.transcribe(
                audio, batch_size=args.batch_size, language="vi", task="transcribe",
                print_progress=False,
            )
            asr_segments = [
                {"start": finite_or_none(seg.get("start")),
                 "end": finite_or_none(seg.get("end")),
                 "text": str(seg.get("text", "")).strip()}
                for seg in asr_result.get("segments", [])
            ]
            asr_text = " ".join(seg["text"] for seg in asr_segments if seg["text"])
            segment_starts = [seg["start"] for seg in asr_segments if seg["start"] is not None]
            segment_ends = [seg["end"] for seg in asr_segments if seg["end"] is not None]
            asr_start = min(segment_starts) if segment_starts else None
            asr_end = max(segment_ends) if segment_ends else None
            reference_words, asr_words = words(transcript), words(asr_text)
            asr_wer_strict = word_error_rate(reference_words, asr_words)
            asr_wer = word_error_rate(
                [fold_vietnamese_word(word) for word in reference_words],
                [fold_vietnamese_word(word) for word in asr_words],
            )
            differences = transcript_diffs(reference_words, asr_words)

        aligned = whisperx.align(
            [{"start": 0.0, "end": duration, "text": transcript}],
            align_model, align_metadata, audio, device,
            return_char_alignments=False, print_progress=False,
        )
        aligned_words = []
        reasons = []
        low_words = []
        for word_index, item in enumerate(aligned.get("word_segments", []), 1):
            start, end = finite_or_none(item.get("start")), finite_or_none(item.get("end"))
            score = finite_or_none(item.get("score"))
            word = str(item.get("word", "")).strip()
            timing_bad = start is None or end is None or end <= start or start < 0 or end > duration + 0.1
            if timing_bad:
                reasons.append("missing_or_invalid_word_timing")
            if args.min_word_score is not None and score is not None and score < args.min_word_score:
                low_words.append({"word": word, "start": start, "end": end, "score": score})
            record = {"index": word_index, "word": word, "start": start,
                      "end": end, "score": score, "timing_review": timing_bad}
            aligned_words.append(record)
            word_rows.append({"file_name": filename, **record})

        aligned_starts = [word["start"] for word in aligned_words if word["start"] is not None]
        aligned_ends = [word["end"] for word in aligned_words if word["end"] is not None]
        aligned_start = min(aligned_starts) if aligned_starts else None
        aligned_end = max(aligned_ends) if aligned_ends else None
        alignment_end_gap = None if asr_end is None or aligned_end is None else max(0.0, asr_end - aligned_end)
        if low_words:
            reasons.append("low_alignment_score")
        if alignment_end_gap is not None and alignment_end_gap > max(0.75, 0.08 * duration):
            reasons.append("forced_alignment_ends_early_vs_asr")
        if asr_wer is not None and asr_wer > args.max_asr_wer:
            reasons.append("transcript_differs_from_asr")
        if not aligned_words:
            reasons.append("no_aligned_words")

        item_result = {
            "file_name": filename, "duration_seconds": round(duration, 3),
            "transcript": transcript, "asr_transcript": asr_text,
            "asr_segments": asr_segments,
            "asr_wer": None if asr_wer is None else round(asr_wer, 4),
            "asr_wer_strict": None if asr_wer_strict is None else round(asr_wer_strict, 4),
            "asr_differences": differences,
            "aligned_start": aligned_start, "aligned_end": aligned_end,
            "asr_start": asr_start, "asr_end": asr_end,
            "alignment_end_gap_vs_asr": alignment_end_gap,
            "alignment_model": args.align_model,
            "aligned_words": aligned_words,
            "low_score_words": low_words,
            "review_recommended": bool(reasons),
            "review_reasons": sorted(set(reasons)),
        }
        result_rows.append(item_result)
        if reasons:
            review_rows.append({
                "file_name": filename, "duration_seconds": round(duration, 3),
                "reasons": ";".join(sorted(set(reasons))),
                "aligned_start": aligned_start, "aligned_end": aligned_end,
                "asr_start": asr_start, "asr_end": asr_end,
                "alignment_end_gap_vs_asr": alignment_end_gap,
                "asr_wer": "" if asr_wer is None else round(asr_wer, 4),
                "asr_wer_strict": "" if asr_wer_strict is None else round(asr_wer_strict, 4),
                "low_score_words": json.dumps(low_words, ensure_ascii=False),
                "transcript": transcript, "asr_transcript": asr_text or "",
                "asr_differences": json.dumps(differences, ensure_ascii=False),
            })
        print(f"[{index}/{len(rows)}] {filename}: {len(aligned_words)} words"
              f" | ASR WER={'n/a' if asr_wer is None else f'{asr_wer:.3f}'}"
              f" | {'REVIEW' if reasons else 'ok'}", flush=True)

    with output_jsonl.open("w", encoding="utf-8") as stream:
        for row in result_rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    with output_words.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("file_name", "index", "word", "start", "end", "score", "timing_review"))
        writer.writeheader()
        writer.writerows(word_rows)
    with output_review.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ("file_name", "duration_seconds", "reasons", "aligned_start", "aligned_end", "asr_start", "asr_end", "alignment_end_gap_vs_asr", "asr_wer", "asr_wer_strict", "low_score_words", "transcript", "asr_transcript", "asr_differences")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review_rows)

    wers = [row["asr_wer"] for row in result_rows if row["asr_wer"] is not None]
    summary = {
        "tool": "WhisperX", "version": __import__('importlib.metadata').metadata.version('whisperx'),
        "language": "vi", "device": device, "compute_type": compute_type,
        "align_model": args.align_model,
        "asr_model": None if args.skip_asr else args.asr_model,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "metadata": str(metadata.resolve()), "audio_dir": str(audio_dir.resolve()),
        "clip_count": len(result_rows),
        "review_clip_count": len(review_rows),
        "mean_asr_wer": None if not wers else round(sum(wers) / len(wers), 4),
        "min_word_score_review_threshold": args.min_word_score,
        "alignment_score_note": "WhisperX word scores are character-path means, not calibrated confidence; thresholding is disabled unless explicitly requested.",
        "max_asr_wer_review_threshold": args.max_asr_wer,
        "outputs": {"word_timings": str(output_words), "full_alignment": str(output_jsonl),
                    "review_list": str(output_review)},
        "policy": "Forced alignment and ASR differences are review cues only; no transcript/audio is edited and no clip is automatically removed or trained.",
    }
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"COMPLETE: {len(result_rows)} clips | {len(review_rows)} recommended for listening review")
    print(f"word timings: {output_words}")
    print(f"review list: {output_review}")
    print(f"summary: {output_summary}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; partial output files, if any, are retained.", file=sys.stderr)
        raise
