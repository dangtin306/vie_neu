"""Generate one TEST8 heads-only WAV for manual listening."""
from pathlib import Path
import csv
import sys
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import test8_nghean_audio_head_only as test8


def main():
    runner = test8.configure(test8.load_runner())
    rows = list(csv.DictReader((test8.DATA_EXP_DIR / "nghean_test_pairs_v2.csv").open(encoding="utf-8-sig", newline="")))
    row = dict(rows[0])
    row["target_text"] = row["target_transcript"]
    row["reference_path"] = Path(row["reference_path"])
    row["target_path"] = Path(row["target_path"])
    engine = runner._load_engine()
    runner.inject(engine)
    state, _ = runner.load_saved_state(test8.EXP_DIR / "best_tf_checkpoint")
    engine.model.load_state_dict(state, strict=False)
    sample = runner.prepare(engine, row)
    codes, _, stop = runner.trace_generation(engine, row["target_transcript"], (sample["speaker_emb"], sample["ref_codes"]), runner.SEED)
    wav = np.asarray(engine._decode_codes(codes), dtype=np.float32).reshape(-1)
    out = test8.EXP_DIR / "quick_demo" / "test8_heads_only_case01.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, wav, 48000)
    print(f"WAV: {out}", flush=True)
    print(f"speaker={row['speakerID']} target={row['target_path'].name} frames={len(codes)} stop={stop} duration={len(wav)/48000:.2f}s", flush=True)


if __name__ == "__main__":
    main()
