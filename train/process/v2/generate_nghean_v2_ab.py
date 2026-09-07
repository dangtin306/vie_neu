"""Generate VieNeu-TTS v2 Nghệ An merged/base A-B audio.

Run with the tts_5 Python environment. This script does inference only;
it does not train, modify source data, or touch the v3 pipeline.
"""

from pathlib import Path
import argparse
import csv
import sys

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "source_code" / "audio_model"
RUN = ROOT / "train" / "output" / "nghean_v2_pilot"
PREPARED = ROOT / "train" / "output" / "nghean_test2_accent_scale"
MERGED = RUN / "merged_model"
BASE = "pnnbao-ump/VieNeu-TTS-0.3B"
DEFAULT_TEXT = "Đây là câu thử nghiệm thứ hai bằng giọng Nghệ An sau khi mô hình đã được huấn luyện."

sys.path.insert(0, str(SOURCE / "src"))


def get_reference():
    manifest = PREPARED / "nghean_eligible_audio_manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        row = next(r for r in rows if r["split"] == "test" and r["speakerID"] == "spk_37_0224")
    return Path(row["local_path"]), row["transcript"]


def generate(repo, ref_audio, ref_text, text, output):
    import torch
    from vieneu import Vieneu

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading: {repo}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    engine = Vieneu(
        mode="standard",
        backbone_repo=str(repo),
        backbone_device=device,
        codec_repo="neuphonic/neucodec",
        codec_device=device,
        gguf_filename=None,
    )
    audio = engine.infer(
        text=text,
        ref_audio=str(ref_audio),
        ref_text=ref_text,
        max_chars=256,
        apply_watermark=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    engine.save(audio, str(output))
    print(f"Saved: {output}")
    engine.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--merged-only", action="store_true")
    a = p.parse_args()

    if not MERGED.exists():
        raise FileNotFoundError(f"Merged model not found: {MERGED}")
    ref_audio, ref_text = get_reference()
    print(f"Reference: {ref_audio}")
    print(f"Text: {a.text}")

    generate(MERGED, ref_audio, ref_text, a.text, RUN / "nghean_merged_test_2.wav")
    if not a.merged_only:
        generate(BASE, ref_audio, ref_text, a.text, RUN / "base_test_2.wav")


if __name__ == "__main__":
    main()
