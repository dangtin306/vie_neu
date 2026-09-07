"""TEST8 inference-only evaluation: test CE and the existing MOSS-DTW metric."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import test8_nghean_audio_head_only as test8  # noqa: E402


def dtw_codes(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    n, m = len(a), len(b)
    if not n or not m:
        return float("nan")
    prev = np.full(m + 1, np.inf, dtype=np.float64)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(m + 1, np.inf, dtype=np.float64)
        for j in range(1, m + 1):
            cost = float(np.mean(a[i - 1] != b[j - 1]))
            cur[j] = cost + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return float(prev[m] / (n + m))


def main():
    runner = test8.configure(test8.load_runner())
    runner.seed(runner.SEED)
    rows = list(csv.DictReader((test8.DATA_EXP_DIR / "nghean_test_pairs_v2.csv").open(encoding="utf-8-sig", newline="")))
    if len(rows) != 9:
        raise RuntimeError(f"expected 9 test pairs, got {len(rows)}")

    base = runner._load_engine()
    tf = runner._load_engine()
    runner.inject(tf)
    state, _ = runner.load_saved_state(test8.EXP_DIR / "best_tf_checkpoint")
    tf.model.load_state_dict(state, strict=False)

    runner.PREP_TOTAL = len(rows) * 2
    result = []
    base_ce = []
    tf_ce = []
    for i, row in enumerate(rows):
        r = dict(row)
        r["target_text"] = r.get("target_transcript", "")
        r["reference_path"] = Path(r["reference_path"])
        r["target_path"] = Path(r["target_path"])
        sb = runner.prepare(base, r)
        st = runner.prepare(tf, r)
        base_loss = float(runner.compute_loss(base, runner.true_history(base, sb), sb["codes"]).detach().cpu())
        tf_loss = float(runner.compute_loss(tf, runner.true_history(tf, st), st["codes"]).detach().cpu())
        base_ce.append(base_loss)
        tf_ce.append(tf_loss)

        target_codes = np.asarray(sb["codes"].detach().cpu())
        for name, engine, sample in (("Base", base, sb), ("TEST8 heads-only", tf, st)):
            codes, _, stop = runner.trace_generation(
                engine,
                row["target_transcript"],
                (sample["speaker_emb"], sample["ref_codes"]),
                runner.SEED + i,
            )
            distance = dtw_codes(target_codes, np.asarray(codes))
            result.append({
                "pair_id": row.get("pair_id", f"pair_{i + 1:02d}"),
                "speakerID": row["speakerID"],
                "model": name,
                "moss_dtw": distance,
                "stop": stop,
            })
            print(f"{i + 1}/9 {name} {row['speakerID']} moss={distance:.6f} stop={stop}", flush=True)

    base_moss = float(np.mean([x["moss_dtw"] for x in result if x["model"] == "Base"]))
    test8_moss = float(np.mean([x["moss_dtw"] for x in result if x["model"] == "TEST8 heads-only"]))
    by_pair = {}
    for x in result:
        by_pair.setdefault(x["pair_id"], {})[x["model"]] = x["moss_dtw"]
    test8_vs_base = sum(v["TEST8 heads-only"] < v["Base"] for v in by_pair.values())
    test8_vs_r8 = None
    old_r8_path = test8.DATA_EXP_DIR / "nghean_target_acoustic_diagnostic.csv"
    if old_r8_path.exists():
        old = list(csv.DictReader(old_r8_path.open(encoding="utf-8-sig", newline="")))
        r8 = {x["pair_id"]: float(x["moss_dtw_distance"]) for x in old if x["model"] == "TF-LoRA"}
        if len(r8) == 9:
            test8_vs_r8 = sum(by_pair[k]["TEST8 heads-only"] < r8[k] for k in r8)

    out = test8.EXP_DIR / "test8_ce_moss_results.json"
    out.write_text(json.dumps({
        "test_ce": {"Base": float(np.mean(base_ce)), "TEST8 heads-only": float(np.mean(tf_ce))},
        "moss_dtw": {"Base": base_moss, "TEST8 heads-only": test8_moss},
        "closer_than_base": test8_vs_base,
        "closer_than_r8": test8_vs_r8,
        "pairs": by_pair,
    }, indent=2), encoding="utf-8")
    print(f"Base test CE={np.mean(base_ce):.6f}; TEST8 test CE={np.mean(tf_ce):.6f}", flush=True)
    print(f"Base MOSS={base_moss:.6f}; TEST8 MOSS={test8_moss:.6f}; closer than Base={test8_vs_base}/9; closer than R8={test8_vs_r8 if test8_vs_r8 is not None else 'N/A'}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
