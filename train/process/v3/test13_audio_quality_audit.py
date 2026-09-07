from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parent.parent
AUDIO_ROOT=ROOT/'vimd_haiphong'
OUT=ROOT/'output'/'test13_scale_feasibility'
SPEAKERS={'spk_15_0218','spk_15_0220','spk_15_0219'}

def silence_metrics(y,sr):
    if y.size==0: return 0.,0.,0.,0
    frame=max(1,int(sr*.02)); n=max(1,int(np.ceil(len(y)/frame)))
    e=np.array([np.sqrt(np.mean(y[i*frame:min(len(y),(i+1)*frame)]**2)) for i in range(n)])
    threshold=max(1e-5,float(np.max(e))*0.02)
    silent=e<=threshold; spans=[]; i=0
    while i<n:
        if not silent[i]: i+=1; continue
        j=i
        while j<n and silent[j]: j+=1
        spans.append((j-i)*frame/sr); i=j
    lead=0.; trail=0.
    if silent.size:
        k=0
        while k<n and silent[k]: k+=1
        lead=min(len(y),k*frame)/sr
        k=n-1
        while k>=0 and silent[k]: k-=1
        trail=min(len(y),max(0,n-1-k)*frame)/sr
    internal=[x for x in spans if x>0 and x < len(y)/sr and x > 0.04]
    total=sum(internal)
    return lead,trail,total/max(1,len(y)/sr),max(internal or [0.])

def metrics(path):
    y,sr=sf.read(path,always_2d=False); y=np.asarray(y,dtype=np.float32)
    if y.ndim>1: y=y.mean(axis=1)
    duration=len(y)/sr if sr else 0.; peak=float(np.max(np.abs(y))) if len(y) else 0.; rms=float(np.sqrt(np.mean(y*y))) if len(y) else 0.; dc=float(np.mean(y)) if len(y) else 0.
    clip=float(np.mean(np.abs(y)>=.999)) if len(y) else 0.
    lead,trail,internal,longest=silence_metrics(y,sr)
    # Conservative automatic flags: these do not identify music/secondary speech.
    status='CLEAN'; notes=[]
    if not len(y) or duration==0: status='BAD'; notes.append('empty/unreadable')
    elif clip>0.001: status='SUSPECT'; notes.append('clipping_ratio>0.1%')
    elif lead>0.8 or trail>0.8 or longest>1.0: status='SUSPECT'; notes.append('long leading/trailing/internal silence')
    return {'duration':duration,'sample_rate':sr,'channels':1 if y.ndim==1 else y.shape[1],'peak':peak,'rms':rms,'dc_offset':dc,'clipping_ratio':clip,'leading_silence':lead,'trailing_silence':trail,'internal_silence_ratio':internal,'longest_internal_silence':longest,'music_flag':'UNASSESSED','noise_flag':'UNASSESSED','secondary_speech_flag':'UNASSESSED','reverb_flag':'UNASSESSED','cut_flag':'UNASSESSED','qc_status':status,'notes':'; '.join(notes)}

def main():
    files=[]
    for p in AUDIO_ROOT.rglob('*.wav'):
        if any(part in SPEAKERS for part in p.parts): files.append(p)
    pair_rows=[]
    for name in ['train_pairs.csv','valid_pairs.csv','test_pairs.csv']:
        with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                if r['speakerID'] in SPEAKERS: pair_rows.append((name[:-10],r))
    used={}
    for split,r in pair_rows:
        for role,key in [('reference','reference_path'),('target','target_path')]:
            used.setdefault(str(Path(r[key]).resolve()),[]).append((split,role,r['speakerID'],Path(r['target_path']).stem))
    rows=[]
    for p in sorted(files):
        spk=next((x for x in SPEAKERS if x in p.parts), 'unknown'); split=next((x for x in ['test','valid','train'] if x in p.parts),'unknown'); key=str(p.resolve()); roles=used.get(key,[('unpaired','unpaired',spk,'')])
        for split2,role,_,pairtarget in roles:
            row={'speakerID':spk,'filename':p.name,'split':split,'role_reference_or_target':role,'pair_target_id':pairtarget,'path':str(p)}; row.update(metrics(p)); rows.append(row)
    fields=list(rows[0]);
    with (OUT/'test13_audio_quality_audit.csv').open('w',encoding='utf-8-sig',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary={'scope':sorted(SPEAKERS),'files_audited':len(files),'rows_with_pair_roles':len(rows),'automatic_audio_flags':'Music/noise/secondary speech/reverb/cut are UNASSESSED without listening or a validated detector.','per_speaker':{}}
    for spk in sorted(SPEAKERS):
        rs=[r for r in rows if r['speakerID']==spk]; summary['per_speaker'][spk]={'files':len({r['path'] for r in rs}),'mean_duration':float(np.mean([r['duration'] for r in rs])) if rs else None,'mean_rms':float(np.mean([r['rms'] for r in rs])) if rs else None,'max_clipping_ratio':max([r['clipping_ratio'] for r in rs] or [0]),'suspect_files':sorted({r['filename'] for r in rs if r['qc_status'] in {'SUSPECT','BAD'}}),'pair_roles':len([r for r in rs if r['role_reference_or_target']!='unpaired'])}
    summary['pair_usage']={}
    for split,r in pair_rows:
        summary['pair_usage'].setdefault(split,[]).append({'speakerID':r['speakerID'],'reference':Path(r['reference_path']).name,'target':Path(r['target_path']).name})
    (OUT/'test13_audio_quality_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'audited {len(files)} files; wrote CSV/JSON to {OUT}')
if __name__=='__main__': main()

