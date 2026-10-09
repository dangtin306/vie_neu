#!/usr/bin/env python3
"""Experimental multi-speaker regional LoRA with a packaged no-WAV voice."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FT = ROOT / "source_code/audio_model/finetune"
SOURCE = ROOT / "train/process/v3/test_4/output/nghean_region_one_lora/dataset_training_smooth_v2"
WORK = HERE / "output/nghean_region_no_anchor"
DATA = WORK / "dataset"
TRAINING = WORK / "training"
RUN_PREFIX = "nghean_region_no_anchor_v5_best_eval"
LATEST_RUN = WORK / "latest_run.json"
BASE = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
VOICE_NAME = "Nghe An - regional prototype"
MANUAL_EXCLUSIONS = {
    "0006_37_0086.wav": "transcript tail appears mismatched/unclear (‘thị tận’)",
    "0008_37_0006.wav": "transcript contains an exact repeated phrase (‘đầu tư, đầu tư’)",
    "0042_37_0235.wav": "uncertain transcript for the loanword ‘sốc hâu’",
    "0050_37_0281.wav": "transcript ends mid-clause with ‘thì’",
}
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--train", action="store_true")
    modes.add_argument("--all", action="store_true")
    p.add_argument("--overwrite-stage", action="store_true",
                   help="replace only test_5's copied dataset")
    p.add_argument("--max-steps", type=int, default=60,
                   help="safe pilot length; prior eval peaked near step 60")
    p.add_argument("--run-name", default=None,
                   help="unique run name; default includes local timestamp")
    p.add_argument("--allow-long-run", action="store_true",
                   help="legacy opt-in flag; test_5 now permits up to 300 steps")
    return p.parse_args()


def stage(overwrite: bool):
    meta = SOURCE / "metadata.csv"
    audio_dir = SOURCE / "raw_audio"
    if not meta.is_file() or not audio_dir.is_dir():
        raise FileNotFoundError(f"Clean source dataset not found: {SOURCE}")
    if DATA.exists():
        if not overwrite:
            raise FileExistsError(f"{DATA} exists; pass --overwrite-stage to replace test_5 data")
        shutil.rmtree(DATA)
    out_audio = DATA / "raw_audio"
    out_audio.mkdir(parents=True)
    rows = []
    with meta.open("r", encoding="utf-8-sig", newline="") as f:
        for line_no, row in enumerate(csv.reader(f, delimiter="|"), 1):
            if len(row) < 2 or not row[0].strip() or not row[1].strip():
                raise ValueError(f"Bad metadata row {line_no}")
            name, text = row[0].strip(), row[1].strip()
            source_wav = audio_dir / name
            if not source_wav.is_file():
                raise FileNotFoundError(source_wav)
            shutil.copy2(source_wav, out_audio / name)
            rows.append((name, text))
    with (DATA / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="|", lineterminator="\n").writerows(rows)
    (WORK / "source_manifest.json").write_text(json.dumps({
        "experiment": "regional multi-speaker LoRA + one packed region speaker prototype",
        "source_dataset": str(SOURCE),
        "clip_count": len(rows),
        "source_speakers": sorted({speaker_key(name) for name, _ in rows}),
        "conditioning": "retain each clip’s upstream 192D speaker embedding for training; package one real in-dataset medoid embedding for no-WAV inference; no reference codes",
        "method_limit": "Uses the components integrated by VieNeu v3 Turbo (SEA-G2P, MOSS audio codec, and its 192D speaker encoder); this multi-speaker regional LoRA remains experimental and cannot guarantee one natural identity.",
        "rows": [{"file_name": n, "speaker_id": speaker_key(n), "text": t} for n, t in rows],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"STAGED: {len(rows)} clips | {len({speaker_key(n) for n, _ in rows})} source speakers")
    print(f"dataset: {DATA}")


def speaker_key(filename: str) -> str:
    parts = Path(filename).stem.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 3 else Path(filename).stem


def run(cmd):
    print("run:", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=ROOT, check=True)


def transcript_flags(text: str):
    # No generic repetition or final-word filters: they reject valid Vietnamese.
    return []

def prepare():
    script = FT / "prepare_dataset.py"
    raw = DATA / "train_per_clip_embeddings.parquet"
    clean = DATA / "train.parquet"
    embedding_file = WORK / "regional_voice_embedding.json"
    run([sys.executable, "-u", script, "--dataset-dir", DATA,
         "--out", raw, "--base", BASE,
         "--min-sec", "1.0", "--max-sec", "20.0"])
    import pyarrow as pa
    import pyarrow.parquet as pq
    source_rows = pq.read_table(str(raw)).to_pylist()
    kept, excluded = [], []
    for row in source_rows:
        name = row["file_name"]
        flags = transcript_flags(row.get("text", ""))
        manual = MANUAL_EXCLUSIONS.get(name)
        reasons = ([manual] if manual else []) + flags
        if reasons:
            excluded.append({"file_name": name, "reasons": reasons})
        else:
            kept.append(row)
    if len(kept) < 12:
        raise RuntimeError(f"Only {len(kept)} clean rows remain; review quality exclusions before training")
    pq.write_table(pa.Table.from_pylist(kept), str(clean))
    select_medoid_embedding(clean, embedding_file)
    audit = {
        "source_rows": len(source_rows), "training_rows": len(kept),
        "excluded_rows": excluded,
        "policy": "keep source text/audio unchanged; exclude only the four manually reviewed clips; no generic repetition or sentence-ending heuristic",
    }
    (WORK / "quality_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = WORK / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["clip_count_training"] = len(kept)
    manifest["excluded_from_training"] = excluded
    manifest["conditioning"] = "per-clip MOSS codes and original upstream 192D speaker embeddings; package a real medoid embedding from retained rows for no-reference inference"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"QUALITY FILTER: {len(kept)}/{len(source_rows)} rows retained")
    for row in excluded:
        print(f"  excluded {row['file_name']}: {'; '.join(row['reasons'])}")
    print(f"PREPARE COMPLETE: {clean} (per-clip conditioning; medoid voice packed)")


def unit(v):
    import numpy as np
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < 1e-8:
        raise ValueError("Invalid/zero speaker embedding")
    return v / n


def select_medoid_embedding(parquet: Path, out: Path):
    """Pack a real in-dataset speaker vector closest to the regional center.

    Training rows retain their individual upstream embeddings. The selected
    medoid is therefore a vector that actually appeared during training, unlike
    an averaged centroid that can sit off the speaker-embedding manifold.
    """
    import numpy as np
    import pyarrow.parquet as pq
    rows = pq.read_table(str(parquet)).to_pylist()
    grouped = defaultdict(list)
    norms = []
    for row in rows:
        emb = np.asarray(row.get("speaker_embedding"), dtype=np.float64).reshape(-1)
        if emb.shape != (192,) or not np.isfinite(emb).all():
            raise ValueError(f"Expected finite 192D speaker embedding in {row.get('file_name')}")
        grouped[speaker_key(row["file_name"])].append((row["file_name"], emb))
        norms.append(float(np.linalg.norm(emb)))
    if len(grouped) < 2:
        raise ValueError("A regional medoid needs multiple source speakers")
    speaker_ids = list(grouped)
    people = []
    for sid in speaker_ids:
        vectors = np.stack([unit(emb) for _, emb in grouped[sid]])
        people.append(unit(vectors.mean(axis=0)))
    people = np.stack(people)
    center = unit(people.mean(axis=0))
    for _ in range(30):
        distances = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (people @ center)))
        med = float(np.median(distances))
        mad = float(np.median(np.abs(distances - med)))
        cutoff = max(med + 2.5 * 1.4826 * mad, 1e-4)
        weights = np.minimum(1.0, cutoff / np.maximum(distances, 1e-8))
        updated = unit((people * weights[:, None]).sum(axis=0))
        if float(np.linalg.norm(updated - center)) < 1e-8:
            center = updated
            break
        center = updated
    scores = people @ center
    medoid_index = int(np.argmax(scores))
    medoid_id = speaker_ids[medoid_index]
    candidates = grouped[medoid_id]
    filename, proto = max(candidates, key=lambda item: float(unit(item[1]) @ people[medoid_index]))
    proto = proto.astype(np.float32)
    result = {
        "name": VOICE_NAME,
        "speaker_emb": [round(float(x), 7) for x in proto],
        "embedding_dim": 192,
        "aggregation": "real in-dataset speaker medoid; exact source embedding, not an averaged vector",
        "clips": len(rows),
        "speakers": len(grouped),
        "medoid_speaker_id": medoid_id,
        "medoid_file": filename,
        "cosine_to_regional_center": round(float(scores[medoid_index]), 5),
        "mean_cosine_to_regional_center": round(float(np.mean(scores)), 5),
        "minimum_cosine_to_regional_center": round(float(np.min(scores)), 5),
        "source_embedding_norm_median": round(float(np.median(norms)), 5),
        "medoid_embedding_norm": round(float(np.linalg.norm(proto)), 5),
        "reference_audio_used_at_inference": False,
        "reference_codes": None,
        "note": "This packages one real speaker embedding from the retained regional dataset, not an averaged synthetic identity.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REGION MEDOID: {len(rows)} clips | {len(grouped)} speakers | {medoid_id} ({filename})")
    print(f"embedding: {out}; cosine to robust region center={result['cosine_to_regional_center']}; mean cosine={result['mean_cosine_to_regional_center']}")


def package_voice(merged: Path, embedding_file: Path):
    data = json.loads(embedding_file.read_text(encoding="utf-8"))
    voice_file = merged / "voices_v3_turbo.json"
    payload = {
        "meta": {"note": "Experimental in-dataset medoid speaker embedding; no reference codes."},
        "default_voice": VOICE_NAME,
        "presets": {VOICE_NAME: {
            "description": "Nghệ An · representative in-dataset speaker embedding",
            "gender": "",
            "speaker_emb": data["speaker_emb"],
            "codes": None,
        }},
    }
    voice_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"PACKED NO-REFERENCE VOICE: {voice_file}")


def train(max_steps: int, run_name: str | None, allow_long_run: bool):
    if max_steps > 300:
        raise ValueError("Refusing >300 steps; the test_5 pilot is capped at 300.")
    parquet = DATA / "train.parquet"
    embfile = WORK / "regional_voice_embedding.json"
    script = FT / "train_lora.py"
    if not parquet.is_file() or not embfile.is_file():
        raise FileNotFoundError("Run --prepare (or --all) first")
    if not run_name:
        run_name = f"{RUN_PREFIX}_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"
    run_dir = TRAINING / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix logs/checkpoints in existing run: {run_dir}; choose a new --run-name")
    run([sys.executable, "-u", script,
         "--data", parquet, "--run", run_name, "--output-dir", TRAINING,
         "--base", BASE, "--subfolder", "update",
         "--r", "16", "--alpha", "32", "--dropout", "0.05",
         "--target", "backbone", "--batch-size", "1", "--grad-accum", "4",
         "--lr", "0.0002", "--max-steps", str(max_steps),
         "--eval-ratio", "0.2", "--eval-every", "5", "--save-every", "5",
         "--log-every", "2", "--max-length", "1024", "--seed", "42",
         "--num-workers", "0"])
    log_path = run_dir / "train_log.jsonl"
    evals = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "eval_loss" in record:
            evals.append(record)
    if not evals:
        raise RuntimeError(f"No evaluation records in {log_path}; refusing to choose a final checkpoint blindly")
    best = min(evals, key=lambda row: (float(row["eval_loss"]), int(row["step"])))
    step = int(best["step"])
    adapter = run_dir / f"checkpoint-{step}"
    if not adapter.is_dir():
        adapter = run_dir / "adapter"  # final adapter exists when best step is the final step
    if not adapter.is_dir():
        raise FileNotFoundError(f"Best adapter missing for eval step {step}: {adapter}")
    merged = run_dir / "merged"
    merge_script = FT / "merge_lora.py"
    run([sys.executable, "-u", merge_script,
         "--adapter", adapter, "--out", merged, "--base", BASE, "--subfolder", "update"])
    if not (merged / "config.json").is_file():
        raise FileNotFoundError(f"Merged model missing: {merged}")
    package_voice(merged, embfile)
    selection = {
        "run_name": run_name, "steps_requested": max_steps,
        "best_eval_step": step, "best_eval_loss": float(best["eval_loss"]),
        "best_eval_acc_cb0": best.get("eval_acc_cb0"),
        "selected_adapter": str(adapter), "merged": str(merged),
    }
    (run_dir / "selected_checkpoint.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    LATEST_RUN.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TRAIN COMPLETE: {run_dir}")
    print(f"BEST CHECKPOINT: step={step} eval_loss={float(best['eval_loss']):.4f}")
    print(f"MERGED NO-ANCHOR MODEL: {merged}")


def main():
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.max_steps > 300:
        raise ValueError("Refusing >300 steps before staging; the test_5 pilot is capped at 300.")
    if args.prepare or args.all:
        stage(args.overwrite_stage)
        prepare()
    if args.train or args.all:
        train(args.max_steps, args.run_name, args.allow_long_run)


if __name__ == "__main__":
    main()
