"""Inference-only Base/TF vs real Nghệ An target acoustic diagnostic."""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT=Path(__file__).resolve().parent.parent; EXP=ROOT/'output'/'nghean_test2_accent_scale'; OUT=EXP/'nghean_target_acoustic_diagnostic.csv'; SEED=20260827
sys.path.insert(0,str(ROOT));
from test2_nghean_accent_scale import (_load_engine, inject, prepare, trace_generation, load_saved_state, BASE_CHECKPOINT, MOSS_REPO)

def dtw_codes(a,b):
    a=np.asarray(a); b=np.asarray(b); n,m=len(a),len(b)
    if not n or not m:return float('nan')
    prev=np.full(m+1,np.inf,dtype=np.float64); prev[0]=0.
    for i in range(1,n+1):
        cur=np.full(m+1,np.inf,dtype=np.float64)
        for j in range(1,m+1): cur[j]=float(np.mean(a[i-1]!=b[j-1]))+min(prev[j],cur[j-1],prev[j-1])
        prev=cur
    return float(prev[m]/(n+m))

def signal_stats(path_or_wav, sr=None):
    if isinstance(path_or_wav,(str,Path)): y,sr=sf.read(path_or_wav,always_2d=False)
    else:y=np.asarray(path_or_wav); sr=int(sr)
    y=np.asarray(y,dtype=np.float32); y=y.mean(1) if y.ndim>1 else y
    dur=len(y)/sr if sr else 0.; frame=max(1,int(sr*.02)); e=np.array([np.sqrt(np.mean(y[i:i+frame]**2)) for i in range(0,len(y),frame)]) if len(y) else np.zeros(1)
    th=max(1e-5,float(e.max())*.02); silent=e<=th; spans=[]; i=0
    while i<len(silent):
        if not silent[i]:i+=1;continue
        j=i
        while j<len(silent) and silent[j]:j+=1
        spans.append((j-i)*frame/sr);i=j
    internal=[x for x in spans if x>.04 and x<dur]
    return {'duration':dur,'silence_ratio':sum(internal)/dur if dur else 0.,'longest_silence':max(internal or [0.])}

def main():
    rows=list(csv.DictReader((EXP/'nghean_test_pairs_v2.csv').open(encoding='utf-8-sig',newline='')))
    if len(rows)!=9: raise RuntimeError(f'expected 9 test pairs, got {len(rows)}')
    base=_load_engine(); tf=_load_engine(); inject(tf); state,_=load_saved_state(EXP/'best_tf_checkpoint'); tf.model.load_state_dict(state,strict=False)
    result=[]
    for i,row in enumerate(rows):
        r=dict(row); r['target_text']=r.get('target_transcript',''); r['reference_path']=Path(r['reference_path']); r['target_path']=Path(r['target_path'])
        sb=prepare(base,r); st=prepare(tf,r); target_codes=np.asarray(sb['codes'].detach().cpu()); target_stats=signal_stats(row['target_path'])
        for name,eng,s in [('Base',base,sb),('TF-LoRA',tf,st)]:
            codes,_,stop=trace_generation(eng,row['target_transcript'],(s['speaker_emb'],s['ref_codes']),SEED+i)
            wav=np.asarray(eng._decode_codes(codes),dtype=np.float32).reshape(-1); gs=signal_stats(wav,48000); dist=dtw_codes(target_codes,np.asarray(codes))
            result.append({'pair_id':row.get('pair_id',f'pair_{i+1:02d}'),'speakerID':row['speakerID'],'model':name,'moss_dtw_distance':dist,'target_duration':target_stats['duration'],'generated_duration':gs['duration'],'duration_error':abs(gs['duration']-target_stats['duration'])/target_stats['duration'] if target_stats['duration'] else float('nan'),'target_silence_ratio':target_stats['silence_ratio'],'silence_error':abs(gs['silence_ratio']-target_stats['silence_ratio']),'target_longest_silence':target_stats['longest_silence'],'longest_silence_error':abs(gs['longest_silence']-target_stats['longest_silence']),'eos':stop=='eos','hit_max':stop=='max_new_frames'})
            print(f'{i+1}/9 {name} {row["speakerID"]} moss={dist:.4f} dur_err={result[-1]["duration_error"]:.3f} stop={stop}',flush=True)
    fields=list(result[0]);
    with OUT.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(result)
    print(f'wrote {OUT}',flush=True)
    for metric in ('moss_dtw_distance','duration_error','silence_error','longest_silence_error'):
        b=np.array([x[metric] for x in result if x['model']=='Base']); t=np.array([x[metric] for x in result if x['model']=='TF-LoRA']); bi=float(b.mean());ti=float(t.mean()); imp=(bi-ti)/bi*100 if bi else 0.; closer=sum(x[metric+'_base'] if False else 0 for x in [])
        print(f'{metric}: Base={bi:.6f} TF={ti:.6f} improvement={imp:.2f}%',flush=True)
    for metric in ('moss_dtw_distance','duration_error','silence_error','longest_silence_error'):
        b={x['pair_id']:x[metric] for x in result if x['model']=='Base'};t={x['pair_id']:x[metric] for x in result if x['model']=='TF-LoRA'}
        print(f'{metric} closer: TF={sum(t[k]<b[k] for k in b)}/9 Base={sum(b[k]<t[k] for k in b)}/9 Tie={sum(b[k]==t[k] for k in b)}/9',flush=True)
if __name__=='__main__':main()
