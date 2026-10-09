#!/usr/bin/env python3
"""Audit training transcripts and generated demos with multilingual Whisper."""
from __future__ import annotations
import csv, json, re, unicodedata
from pathlib import Path
import torch, whisper

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / "train/process/v3/test_4/output/nghean_region_one_lora"
SOURCE = WORK / "dataset"
OUTS = [
    WORK / "demo_compare_previous_90/demo_report.json",
    WORK / "demo_compare_trainclean_90/demo_report.json",
    WORK / "demo_compare_smooth_v2_45/demo_report.json",
    WORK / "demo_compare_smooth_v2_90/demo_report.json",
]
CACHE = ROOT / "application/huggingface/whisper"
REPORT = WORK / "whisper_audit.json"

def norm(text: str) -> list[str]:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z0-9]+", text)

def wer(ref: str, hyp: str) -> float:
    a, b = norm(ref), norm(hyp)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j-1] + (x != y)))
        prev = cur
    return prev[-1] / max(1, len(a))

def repeats(text: str):
    w = re.findall(r"[^\W\d_]+", unicodedata.normalize("NFC", text.lower()), flags=re.UNICODE)
    found = []
    for n in range(1, 6):
        for i in range(len(w) - 2*n + 1):
            if w[i:i+n] == w[i+n:i+2*n]:
                phrase = " ".join(w[i:i+n])
                if not any(x["phrase"] == phrase for x in found):
                    found.append({"phrase": phrase, "token_index": i})
    return found

def transcribe(model, path: Path):
    result = model.transcribe(str(path), language="vi", task="transcribe", fp16=True,
                              temperature=0, condition_on_previous_text=False,
                              verbose=False)
    return result["text"].strip()

def main():
    torch.set_num_threads(2)
    model = whisper.load_model("small", device="cuda", download_root=str(CACHE))
    rows = []
    with (SOURCE / "metadata.csv").open(encoding="utf-8-sig", newline="") as f:
        source_rows = [(r[0].strip(), r[1].strip()) for r in csv.reader(f, delimiter="|")
                       if len(r) >= 2 and r[0].strip() and r[1].strip()]
    for filename, expected in source_rows:
        p = SOURCE / "raw_audio" / filename
        asr = transcribe(model, p)
        item = {"kind": "source", "file": filename, "expected": expected,
                "whisper": asr, "wer": round(wer(expected, asr), 4),
                "asr_repeats": repeats(asr), "expected_repeats": repeats(expected)}
        rows.append(item)
        print(f"SOURCE {filename} WER={item['wer']:.3f} repeats={item['asr_repeats']}\n  ASR: {asr}")
    for report_path in OUTS:
        if not report_path.is_file():
            continue
        demo = json.loads(report_path.read_text(encoding="utf-8"))
        result = demo["results"][0]
        p = Path(result["wav"])
        expected = result["text"]
        asr = transcribe(model, p)
        item = {"kind": "demo", "file": str(p), "expected": expected,
                "whisper": asr, "wer": round(wer(expected, asr), 4),
                "asr_repeats": repeats(asr)}
        rows.append(item)
        print(f"DEMO {report_path.parent.name} duration={result['duration_sec']} WER={item['wer']:.3f} repeats={item['asr_repeats']}\n  ASR: {asr}")
    REPORT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REPORT: {REPORT}")

if __name__ == "__main__":
    main()
