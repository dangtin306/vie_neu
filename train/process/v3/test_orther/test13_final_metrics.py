"""Build final TEST13 reports from existing outputs; never trains or changes models."""
from __future__ import annotations
import csv, json, math, random, statistics
from pathlib import Path
import numpy as np
import soundfile as sf

try:
    import librosa
except Exception:
    librosa = None

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "test13_scale_feasibility"
SUMMARY = OUT / "summary.json"
PAIR_FILES = {"train": ROOT / "train_pairs.csv", "valid": ROOT / "valid_pairs.csv", "test": ROOT / "test_pairs.csv"}
MODELS = ["base", "tf", "warm98_2"]
SEED = 20250826

def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def audio_features(path):
    y, sr = sf.read(path, always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1: y = y.mean(axis=1)
    if not len(y): return {"sample_rate": sr, "rms_mean": None, "rms_std": None, "voiced_ratio": None, "f0_mean": None, "f0_median": None, "f0_std": None, "f0_p05": None, "f0_p95": None}
    rms = np.sqrt(np.maximum(1e-12, librosa.feature.rms(y=y, frame_length=min(2048, len(y)), hop_length=max(128, min(512, len(y)//4 or 128)))[0])) if librosa else np.array([np.sqrt(np.mean(y*y))])
    result = {"sample_rate": sr, "rms_mean": float(np.mean(rms)), "rms_std": float(np.std(rms)), "voiced_ratio": None, "f0_mean": None, "f0_median": None, "f0_std": None, "f0_p05": None, "f0_p95": None}
    if librosa and len(y) >= 2048:
        f0 = librosa.yin(y, fmin=60, fmax=500, sr=sr, frame_length=2048, hop_length=256)
        v = f0[np.isfinite(f0) & (f0 > 0)]
        result["voiced_ratio"] = float(len(v) / max(1, len(f0)))
        if len(v):
            result.update(f0_mean=float(np.mean(v)), f0_median=float(np.median(v)), f0_std=float(np.std(v)), f0_p05=float(np.percentile(v, 5)), f0_p95=float(np.percentile(v, 95)))
    return result

def find_wav(model, i, j):
    hits = sorted((OUT / "blind_audio" / model).glob(f"case_{i:02d}_{j:02d}_*.wav"))
    if not hits: raise FileNotFoundError(f"missing {model} case {i:02d}/{j:02d}")
    return hits[-1]

def flatten_free(summary, model):
    rows=[]
    for sp, speakers in summary[model]["free_run"]["cases"].items():
        for pair, sentences in speakers.items():
            for sid, v in sentences.items():
                rows.append((sp, pair, sid, v))
    return rows

def stats(values):
    vals=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not vals: return {"mean": None, "median": None, "std": None, "worst": None}
    return {"mean": float(np.mean(vals)), "median": float(np.median(vals)), "std": float(np.std(vals)), "worst": float(max(vals))}

def main():
    s=json.loads(SUMMARY.read_text(encoding="utf-8"))
    test=read_rows(PAIR_FILES["test"])
    target_by_pair={f"pair_{i:02d}_{Path(r['target_path']).stem}": r for i,r in enumerate(test)}
    metric_rows=[]
    for model in MODELS:
        for i,(sp,pair,sid,v) in enumerate(flatten_free(s,model)):
            pair_i=int(pair.split("_")[1]); j=int(sid.rsplit("_",1)[1])-1
            wav_path=find_wav(model,pair_i,j)
            feat=audio_features(wav_path)
            target=target_by_pair.get(pair)
            # TEST13 free-run uses fixed SENTENCES, not target_text; target duration is
            # therefore not comparable and must not be used as a quality metric here.
            target_d=0
            gen_d=float(v["duration_sec"])
            metric_rows.append({"speakerID":sp,"pair_id":pair,"model":model,"sentence_id":sid,"teacher_forced_CE":None,"generated_frames":v["frames"],"duration_sec":gen_d,"target_duration_sec":None,"duration_ratio":None,"absolute_duration_error_sec":None,"EOS_success":v["eos"],"hit_max_new_frames":v["hit_max"],"silence_ratio":v["silence_ratio"],"internal_silence_count":v["internal_silence_count"],"longest_internal_silence_sec":v["longest_silence_sec"],"silence_over_300ms":v["silence_over_300ms"],"silence_over_500ms":v["silence_over_500ms"],"repeated_frame_ratio":v["repeated_frame_ratio"],**feat,"speaker_cosine":None,"CER":None,"WER":None,"wav_path":str(wav_path)})
    fields=list(metric_rows[0])
    with (OUT/"test13_final_metrics.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(metric_rows)
    summary_rows=[]
    for model in MODELS:
        rs=[r for r in metric_rows if r["model"]==model]
        ce=s[model].get("test_ce")
        for metric in ["duration_sec","duration_ratio","absolute_duration_error_sec","generated_frames","silence_ratio","longest_internal_silence_sec","repeated_frame_ratio","voiced_ratio","f0_mean","f0_median","f0_std","f0_p05","f0_p95","rms_mean","rms_std","speaker_cosine","CER","WER"]:
            z=stats([r[metric] for r in rs]); summary_rows.append({"model":model,"metric":metric,"mean":z["mean"],"median":z["median"],"std":z["std"],"worst":z["worst"],"test_CE":ce if metric=="duration_sec" else None})
        summary_rows += [{"model":model,"metric":"EOS_success_pct","mean":100*sum(r["EOS_success"] for r in rs)/len(rs),"median":None,"std":None,"worst":None,"test_CE":None},{"model":model,"metric":"hit_max_pct","mean":100*sum(r["hit_max_new_frames"] for r in rs)/len(rs),"median":None,"std":None,"worst":None,"test_CE":None}]
    with (OUT/"test13_model_summary.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["model","metric","mean","median","std","worst","test_CE"]); w.writeheader(); w.writerows(summary_rows)
    random.seed(SEED); mapping={}; sheet=[]
    for i in range(len(test)):
        for j in range(3):
            keys=["base","tf","warm98_2"]; random.shuffle(keys); pid=f"pair_{i:02d}_{Path(test[i]['target_path']).stem}_{j+1:02d}"; mapping[pid]={chr(65+k):str(find_wav(m,i,j)) for k,m in enumerate(keys)}
            row={"pair_id":pid,"speakerID":test[i]["speakerID"]}
            for c in ["accent","naturalness","rhythm","pronunciation","speaker_similarity"]:
                for x in "ABC": row[f"{x}_score_{c}"]=None
            row.update(accent_winner=None,naturalness_winner=None,notes=None); sheet.append(row)
    (OUT/"test13_blind_mapping.json").write_text(json.dumps(mapping,ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT/"test13_blind_score_sheet.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(sheet[0])); w.writeheader(); w.writerows(sheet)
    final={"source_summary":str(SUMMARY),"cases_per_model":33,"models":MODELS,"test_ce":{m:s[m]["test_ce"] for m in MODELS},"validation_ce":{m:s[m]["validation_ce"] for m in MODELS},"speaker_cosine":"N/A (no reliable speaker encoder available)","CER_WER":"N/A (no reliable ASR available)","blind_listening":"PENDING manual scoring","outputs":["test13_final_metrics.csv","test13_model_summary.csv","test13_blind_score_sheet.csv","test13_blind_mapping.json"]}
    (OUT/"test13_final_summary.json").write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"wrote TEST13 final reports: {OUT}")
if __name__=="__main__": main()
