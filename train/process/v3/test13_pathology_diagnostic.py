"""TEST13 pathology diagnostic: inference only, no backward/optimizer/training."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np, torch, soundfile as sf

ROOT=Path(__file__).resolve().parent.parent; SRC=ROOT.parent/'source_code'/'audio_model'/'src'; OUT=ROOT/'output'/'test13_scale_feasibility'; DIAG=OUT/'pathology_diagnostic'
sys.path.insert(0,str(SRC)); sys.path.insert(0,str(ROOT))
import test13_scale_feasibility as t13

MODELS={'Base':None,'TF-LoRA':OUT/'tf_best','Warm98_2':OUT/'warm98_best'}
TEXTS=[('error_01','Hôm nay thời tiết khá dễ chịu.'),('short_01','Mai gặp nhé.'),('medium_01','Chiều nay chúng ta sẽ gặp nhau ở đâu?'),('long_01','Tôi vừa hoàn thành công việc và đang chuẩn bị về nhà, sau đó sẽ ghé qua đón bạn.'),('stable_01','Cảm ơn bạn, hẹn gặp lại vào ngày mai.')]

def load_base(): return t13._load_engine()
def add_lora(engine, path):
    t13.load_adapter(engine,path); return engine
def prep_ref(engine,row):
    return t13.prepare(engine,row)
def metric(wav,codes,stop,spk_norm,xvec_norm):
    m=t13.waveform_metrics(np.asarray(wav,dtype=np.float32).reshape(-1),codes)
    return {'duration':len(wav)/48000.,'frames':len(codes),'EOS':stop=='eos','hit_max':stop=='max_new_frames','silence_ratio':m['silence_ratio'],'longest_silence':m['longest_silence_sec'],'speaker_embedding_norm':spk_norm,'xvec_proj_norm':xvec_norm}

def main():
    rows=list(csv.DictReader((ROOT/'test_pairs.csv').open(encoding='utf-8-sig',newline='')))
    # 0218 has two distinct references; 0220 and 0219 are controls.
    chosen={r['speakerID']:r for r in rows if r['speakerID'] in {'spk_15_0218','spk_15_0220','spk_15_0219'}}
    ref_rows=[next(r for r in rows if r['speakerID']=='spk_15_0218' and Path(r['reference_path']).stem=='15_0294'),next(r for r in rows if r['speakerID']=='spk_15_0218' and Path(r['reference_path']).stem=='15_0295'),chosen['spk_15_0220'],chosen['spk_15_0219']]
    seed=20260827; all_rows=[]; DIAG.mkdir(parents=True,exist_ok=True)
    base=load_base(); refs={}
    for r in ref_rows:
        s=prep_ref(base,r); spk=np.asarray(s['speaker_emb']); spk_t=torch.as_tensor(spk,dtype=torch.float32,device=base.device)
        if base.model.xvec_proj is not None:
            xdtype=next(base.model.xvec_proj.parameters()).dtype
            xp=float(base.model.xvec_proj(spk_t.to(dtype=xdtype)).float().norm().detach().cpu())
        else: xp=None
        key=Path(r['reference_path']).stem; refs[key]=(s['row']['speakerID'],s['speaker_emb'],s['ref_codes'],float(np.linalg.norm(spk)),xp)
    # reference swap: same 0218, fixed error text
    jobs=[('reference_swap','ref_0294','15_0294','error_01'),('reference_swap','ref_0295','15_0295','error_01')]
    # text sweep on ref 0294
    jobs += [('text_sweep','ref_0294','15_0294',tid) for tid,_ in TEXTS]
    # cross-speaker control on same error text
    jobs += [('cross_speaker',f"ref_{Path(r['reference_path']).stem}",Path(r['reference_path']).stem,'error_01') for r in ref_rows[1:]]
    jobs=[(k,label,refkey.replace('ref_',''),tid) for k,label,refkey,tid in jobs]
    for model,path in MODELS.items():
        e=load_base()
        if path: add_lora(e,path)
        for kind,label,refkey,tid in jobs:
            text=dict(TEXTS)[tid]; spk,spkemb,refcodes,spkn,xpn=refs[refkey]
            codes,_,stop=t13.trace_generation(e,text,(spkemb,refcodes),seed)
            wav=np.asarray(e._decode_codes(codes),dtype=np.float32).reshape(-1)
            case=f'{kind}_{label}_{tid}_{model.replace("-","")}.wav'; sf.write(DIAG/case,wav,48000)
            z=metric(wav,codes,stop,spkn,xpn); all_rows.append({'speaker':spk,'reference':refkey,'text_id':tid,'model':model,**z,'wav_path':str(DIAG/case)})
        del e; torch.cuda.empty_cache()
    fields=list(all_rows[0]);
    with (OUT/'test13_pathology_matrix.csv').open('w',encoding='utf-8-sig',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(all_rows)
    # Aggregate concise evidence by speaker/reference/text/model.
    summary={'design':{'models':list(MODELS),'sampling':'TEST13 temperature=.8 top_k=25 top_p=.95 repetition_penalty=1.2 max_new_frames=300','seed':seed},'findings':{},'speaker_conditioning':{}}
    for model in MODELS:
        rs=[r for r in all_rows if r['model']==model]; summary['findings'][model]={'n':len(rs),'mean_duration':float(np.mean([r['duration'] for r in rs])),'mean_silence':float(np.mean([r['silence_ratio'] for r in rs])),'eos':sum(r['EOS'] for r in rs),'hit_max':sum(r['hit_max'] for r in rs)}
    for key,v in refs.items(): summary['speaker_conditioning'][key]={'speaker':v[0],'speaker_embedding_norm':v[3],'xvec_proj_norm':v[4]}
    summary['interpretation']='Automatic metrics only; no accent or intelligibility conclusion.'
    (OUT/'test13_pathology_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'wrote {OUT/"test13_pathology_matrix.csv"}'); print(f'wrote {OUT/"test13_pathology_summary.json"}')
if __name__=='__main__': main()
