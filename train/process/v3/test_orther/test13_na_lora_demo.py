"""TEST13: scaled HaiPhong paired-data feasibility run for v3 Turbo LoRA r=8."""
from __future__ import annotations
import csv, hashlib, json, random, shutil, sys, time, os
from pathlib import Path
import numpy as np
import torch

TRAIN_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = TRAIN_DIR.parent
SOURCE_SRC = PROJECT_DIR / "source_code" / "audio_model" / "src"
OUT = TRAIN_DIR / "output" / "test13_na_lora_demo"
LOG_FILE = OUT / "run.log"
CACHE_DIR = OUT / "cache"
BASE_CHECKPOINT = "pnnbao-ump/VieNeu-TTS-v3-Turbo/update"
MOSS_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
TF_STEPS, WARM_STEPS, LR, WARM_LR = 100, 40, 1e-4, 5e-5
SEED = 20260827
TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY, MAX_FRAMES = .8, 25, .95, 1.2, 300
SENTENCES = [
    ("target_01", "Hôm nay thời tiết khá dễ chịu."),
    ("target_02", "Chiều nay chúng ta sẽ gặp nhau ở đâu?"),
    ("target_03", "Tôi vừa hoàn thành công việc và đang chuẩn bị về nhà."),
]
REF_CACHE = {}
CODE_CACHE = {}
PHONE_CACHE = {}
PREP_DONE = 0
PREP_TOTAL = 0
sys.path.insert(0, str(SOURCE_SRC)); sys.path.insert(0, str(TRAIN_DIR))
from lora_one_sample_ffn_test import inject_ffn_lora  # noqa: E402
from overfit_one_sample_test import build_prompt, compute_loss, precompute_h  # noqa: E402
from test10_mixed_history_pilot import load_engine as _load_engine, mixed_history_hs, lora_state  # noqa: E402
from test7_pause_rhythm_diagnosis import waveform_metrics  # noqa: E402
from test8_free_running_trace import trace_generation  # noqa: E402

def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

class Tee:
    def __init__(self, *streams): self.streams=streams
    def write(self, data):
        for stream in self.streams:
            stream.write(data); stream.flush()
    def flush(self):
        for stream in self.streams: stream.flush()

def read_pairs(name):
    with (TRAIN_DIR / name).open("r", encoding="utf-8-sig", newline="") as f: rows = list(csv.DictReader(f))
    for r in rows:
        r["reference_path"] = Path(r["reference_path"]); r["target_path"] = Path(r["target_path"])
        if r["reference_path"] == r["target_path"] or not r["reference_path"].is_file() or not r["target_path"].is_file() or not r["target_text"].strip(): raise RuntimeError(f"invalid pair: {r}")
    return rows

def validate_split(train, valid, test):
    sets = {k: {r["speakerID"] for r in v} for k,v in {"train":train,"valid":valid,"test":test}.items()}
    if sets["train"] & sets["valid"] or sets["train"] & sets["test"] or sets["valid"] & sets["test"]: raise RuntimeError("speaker overlap in pairs")
    return sets

def prepare(engine, row):
    global PREP_DONE
    ref_key = str(row["reference_path"])
    target_key = str(row["target_path"])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ref_file = Path(ref_key); target_file = Path(target_key)
    def cache_key(path, kind):
        st = path.stat()
        raw = f"{kind}|{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|moss={MOSS_REPO}|base={BASE_CHECKPOINT}".encode()
        return hashlib.sha256(raw).hexdigest()
    ref_cache = CACHE_DIR / f"reference_{cache_key(ref_file, 'reference')}.pt"
    target_cache = CACHE_DIR / f"target_{cache_key(target_file, 'target')}.pt"
    if ref_key not in REF_CACHE:
        if ref_cache.exists():
            saved = torch.load(ref_cache, map_location="cpu", weights_only=True)
            REF_CACHE[ref_key] = (saved["speaker_emb"].numpy(), saved["ref_codes"].numpy())
            print(f"cache hit reference: {ref_file.name}", flush=True)
        else:
            spk0, ref0 = engine.prepare_reference(ref_key, denoise=False, use_ref_codes=True)
            REF_CACHE[ref_key] = (spk0, np.asarray(ref0))
            torch.save({"speaker_emb": torch.as_tensor(spk0).cpu(), "ref_codes": torch.as_tensor(ref0).cpu()}, ref_cache)
            print(f"cache miss reference -> saved: {ref_file.name}", flush=True)
    spk, ref = REF_CACHE[ref_key]
    if target_key not in CODE_CACHE:
        if target_cache.exists():
            CODE_CACHE[target_key] = torch.load(target_cache, map_location="cpu", weights_only=True).to(engine.device)
            print(f"cache hit target: {target_file.name}", flush=True)
        else:
            wav, sr = engine._load_mono(target_key, None)
            CODE_CACHE[target_key] = torch.as_tensor(engine._encode_ref_wav(wav, sr), dtype=torch.long, device=engine.device)
            torch.save(CODE_CACHE[target_key].detach().cpu(), target_cache)
            print(f"cache miss target -> saved: {target_file.name}", flush=True)
    codes = CODE_CACHE[target_key]
    text_key = hashlib.sha256((row["target_text"] + "|phonemizer=v3turbo").encode("utf-8")).hexdigest()
    phone_file = CACHE_DIR / f"phonemes_{text_key}.pt"
    if text_key not in PHONE_CACHE:
        if phone_file.exists():
            PHONE_CACHE[text_key] = torch.load(phone_file, map_location="cpu", weights_only=True)
        else:
            from vieneu_utils.phonemize_text import phonemize_text_with_emotions
            PHONE_CACHE[text_key] = phonemize_text_with_emotions(row["target_text"])
            torch.save(PHONE_CACHE[text_key], phone_file)
    prompt = engine._build_prompt_2d(PHONE_CACHE[text_key], None, np.asarray(ref), engine._resolve_style_id())
    PREP_DONE += 1
    print(f"prepare {PREP_DONE}/{PREP_TOTAL}: {row['split']} {row['speakerID']} {target_file.name}", flush=True)
    return {"row":row, "speaker_emb":spk, "ref_codes":np.asarray(ref), "codes":codes, "prompt":prompt}

def true_history(engine, sample):
    return precompute_h(engine, sample["prompt"], sample["codes"], sample["speaker_emb"])

def inject(engine):
    for p in engine.model.parameters(): p.requires_grad = False
    engine.model.acoustic_decoder.float(); engine.model.audio_embeddings.float(); engine.model.audio_lm_heads.float()
    if engine.model.xvec_proj is not None: engine.model.xvec_proj.float()
    inject_ffn_lora(engine.model.acoustic_decoder.layers[0], 8, 16., 0.)
    for n,p in engine.model.named_parameters(): p.requires_grad = ".lora_A" in n or ".lora_B" in n

def save_adapter(engine, path, step, val, mode):
    path.mkdir(parents=True, exist_ok=True); torch.save(lora_state(engine.model), path/"adapter_model.pt")
    (path/"adapter_config.json").write_text(json.dumps({"rank":8,"alpha":16,"dropout":0.,"best_step":step,"validation_ce":val,"mode":mode},indent=2),encoding="utf-8")

def load_adapter(engine, path):
    inject(engine); engine.model.load_state_dict(torch.load(path/"adapter_model.pt",map_location="cpu",weights_only=True),strict=False)

def load_saved_state(path):
    cfg = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    return torch.load(path / "adapter_model.pt", map_location="cpu", weights_only=True), cfg

def evaluate(engine, samples):
    losses=[]
    with torch.no_grad():
        for i,s in enumerate(samples, 1):
            loss=compute_loss(engine,true_history(engine,s),s["codes"])
            losses.append(loss)
            print(f"evaluate {i}/{len(samples)}: {s['row']['split']} {s['row']['speakerID']} {s['row']['target_path'].name} ce={float(loss.cpu()):.6f}", flush=True)
    return float(torch.stack(losses).mean().cpu())

def train_stage(engine, train, valid, steps, lr, mode):
    opt=torch.optim.AdamW([p for p in engine.model.parameters() if p.requires_grad],lr=lr,weight_decay=0.)
    curve=[]; best=(float("inf"),0); best_state=None
    for step in range(1,steps+1):
        s=train[(step-1)%len(train)]; opt.zero_grad(set_to_none=True)
        hs = true_history(engine,s) if mode=="tf" else mixed_history_hs(engine,s,.02,SEED+step)
        loss=compute_loss(engine,hs,s["codes"])
        if not torch.isfinite(loss): raise RuntimeError(f"NaN/Inf loss step {step}")
        loss.backward(); grad=torch.nn.utils.clip_grad_norm_([p for p in engine.model.parameters() if p.requires_grad],1.)
        if not torch.isfinite(torch.as_tensor(grad)): raise RuntimeError(f"NaN/Inf gradient step {step}")
        opt.step()
        if step%20==0 or step==steps:
            val=evaluate(engine,valid); item={"step":step,"train_ce":float(loss.detach().cpu()),"val_ce":val,"grad_norm":float(torch.as_tensor(grad))}; curve.append(item); print(f"{mode} step {step}: train={item['train_ce']:.6f} val={val:.6f} grad={item['grad_norm']:.6f}",flush=True)
            if val<best[0]: best=(val,step); best_state=lora_state(engine.model)
    return curve,best,best_state

def free_eval(engine, samples, tag):
    out={}; blind=OUT/"blind_audio"/tag; blind.mkdir(parents=True,exist_ok=True)
    total=len(samples)*len(SENTENCES); done=0
    for i,s in enumerate(samples):
        sp=s["row"]["speakerID"]
        pair_key=f"pair_{i:02d}_{s['row']['target_path'].stem}"
        out.setdefault(sp,{})[pair_key]={}
        for j,(sid,text) in enumerate(SENTENCES):
            codes,_,stop=trace_generation(engine,text,(s["speaker_emb"],s["ref_codes"]),SEED+j)
            wav=np.asarray(engine._decode_codes(codes),dtype=np.float32).reshape(-1); m=waveform_metrics(wav,codes)
            name=f"case_{i:02d}_{j:02d}_{random.randint(100000,999999)}.wav"; import soundfile as sf; sf.write(blind/name,wav,48000)
            out[sp][pair_key][sid]={"frames":len(codes),"duration_sec":len(wav)/48000.,"stop":stop,"eos":stop=="eos","hit_max":stop=="max_new_frames",**{k:v for k,v in m.items() if k not in {"duration_sec","generated_frames"}}}
            done+=1
            print(f"free_eval {tag} {done}/{total}: pair {i+1}/{len(samples)} {sp} {s['row']['target_path'].name} {sid} frames={len(codes)} stop={stop}", flush=True)
    flat=[x for sp in out.values() for pair in sp.values() for x in pair.values()]
    return {"cases":out,"aggregate":{"duration_mean":float(np.mean([x["duration_sec"] for x in flat])),"frames_mean":float(np.mean([x["frames"] for x in flat])),"eos_success":int(sum(x["eos"] for x in flat)),"total":len(flat),"hit_max":int(sum(x["hit_max"] for x in flat)),"silence_ratio_mean":float(np.mean([x["silence_ratio"] for x in flat])),"longest_silence_mean":float(np.mean([x["longest_silence_sec"] for x in flat])),"repeated_frame_ratio_mean":float(np.mean([x["repeated_frame_ratio"] for x in flat])),"over_10s":int(sum(x["duration_sec"]>10 for x in flat)),"over_15s":int(sum(x["duration_sec"]>15 for x in flat)),"silence_over_500ms":int(sum(x["silence_over_500ms"] for x in flat))}}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log_handle=LOG_FILE.open("a", encoding="utf-8", buffering=1)
    sys.stdout=Tee(sys.__stdout__,log_handle); sys.stderr=Tee(sys.__stderr__,log_handle)
    print(f"START test13_na_lora_demo pid={os.getpid()} device_probe={torch.cuda.is_available()}", flush=True)
    OUT.mkdir(parents=True,exist_ok=True); seed(SEED)
    train_rows,valid_rows,test_rows=read_pairs("na_train_pairs.csv"),read_pairs("na_valid_pairs.csv"),read_pairs("na_valid_pairs.csv"); sets=validate_split(train_rows,valid_rows,[])
    print("stage: loading engine", flush=True)
    global PREP_TOTAL
    PREP_TOTAL = len(train_rows) + len(valid_rows) + len(test_rows)
    print(f"pairs train/valid/test: {len(train_rows)}/{len(valid_rows)}/{len(test_rows)}; speakers: {len(sets['train'])}/{len(sets['valid'])}/{len(sets['test'])}",flush=True)
    engine=_load_engine(); inject(engine)
    train=[prepare(engine,r) for r in train_rows]; valid=[prepare(engine,r) for r in valid_rows]; test=[prepare(engine,r) for r in test_rows]
    base_val=evaluate(engine,valid); base_test=evaluate(engine,test); print(f"Base validation CE={base_val:.6f}; test CE={base_test:.6f}",flush=True)
    tf_dir = OUT / "tf_best"
    if (tf_dir / "adapter_model.pt").exists() and (tf_dir / "adapter_config.json").exists():
        tf_state, tf_cfg = load_saved_state(tf_dir); tf_best = (float(tf_cfg["validation_ce"]), int(tf_cfg["best_step"])); tf_curve = []; print(f"resume: loaded TF checkpoint step {tf_best[1]}", flush=True)
    else:
        tf_curve,tf_best,tf_state=train_stage(engine,train,valid,TF_STEPS,LR,"tf"); save_adapter(engine,tf_dir,tf_best[1],tf_best[0],"teacher_forcing")
    engine.model.load_state_dict(tf_state,strict=False); tf_val=evaluate(engine,valid); tf_test=evaluate(engine,test)
    warm_dir = OUT / "warm98_best"
    if (warm_dir / "adapter_model.pt").exists() and (warm_dir / "adapter_config.json").exists():
        warm_state, warm_cfg = load_saved_state(warm_dir); warm_best = (float(warm_cfg["validation_ce"]), int(warm_cfg["best_step"])); warm_curve = []; print(f"resume: loaded warm98/2 checkpoint step {warm_best[1]}", flush=True)
    else:
        warm_curve,warm_best,warm_state=train_stage(engine,train,valid,WARM_STEPS,WARM_LR,"warm98_2"); save_adapter(engine,warm_dir,warm_best[1],warm_best[0],"warm98_2")
    engine.model.load_state_dict(warm_state,strict=False); warm_val=evaluate(engine,valid); warm_test=evaluate(engine,test)
    results={"config":{"base":BASE_CHECKPOINT,"lora_rank":8,"tf_steps":TF_STEPS,"warm_steps":WARM_STEPS,"train_pairs":len(train),"valid_pairs":len(valid),"test_pairs":len(test)},"base":{"validation_ce":base_val,"test_ce":base_test},"tf":{"curve":tf_curve,"best_step":tf_best[1],"validation_ce":tf_val,"test_ce":tf_test},"warm98_2":{"curve":warm_curve,"best_step":warm_best[1],"validation_ce":warm_val,"test_ce":warm_test}}
    # Recreate engines for clean Base/TF/Warm inference and use test references.
    for tag,state in (("base",None),("tf",tf_state),("warm98_2",warm_state)):
        e=_load_engine();
        if state is not None: inject(e); e.model.load_state_dict(state,strict=False)
        ts=[prepare(e,r) for r in test_rows]; results[tag]["free_run"]=free_eval(e,ts,tag)
    (OUT/"summary.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8"); print(f"summary: {OUT/'summary.json'}")

if __name__=="__main__": main()
