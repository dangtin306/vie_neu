from __future__ import annotations
import csv,json
from collections import defaultdict
from pathlib import Path
import fsspec, pyarrow.parquet as pq
import soundfile as sf
import numpy as np

ROOT=Path(__file__).resolve().parent; AUDIT=ROOT/'vimd_na_audit.json'; OUT=ROOT/'vimd_nghean'; SHARDS={'train':103,'valid':13,'test':14}; META=['province_name','province_code','filename','text','speakerID','gender']; MAX_FILES=100
def url(s,i): return f'https://huggingface.co/datasets/nguyendv02/ViMD_Dataset/resolve/main/data/{s}-{i:05d}-of-{SHARDS[s]:05d}.parquet'
def select(audit):
 by=defaultdict(list)
 for spk,rs in audit['speaker_rows'].items():
  if len(rs)>=2:
   for r in sorted(rs,key=lambda x:(x['split'],x['filename'])): by[spk].append(r)
 chosen=[]
 # Round-robin across eligible speakers to avoid selecting one speaker's clips only.
 for turn in range(max((len(v) for v in by.values()),default=0)):
  for spk in sorted(by):
   if turn<len(by[spk]) and len(chosen)<MAX_FILES: chosen.append(by[spk][turn])
 return {(r['split'],r['filename']):r for r in chosen}
def stats(p):
 try:
  y,sr=sf.read(p,always_2d=False); y=np.asarray(y,dtype=np.float32); ch=1 if y.ndim==1 else y.shape[1]; y=y.mean(axis=1) if y.ndim>1 else y; peak=float(np.max(np.abs(y))) if len(y) else 0.; rms=float(np.sqrt(np.mean(y*y))) if len(y) else 0.; dur=float(len(y)/sr) if sr else 0.; reasons=[]
  if not 4<=dur<=15: reasons.append('duration_outside_4_15s')
  if peak>=.999: reasons.append('possible_clipping')
  return {'duration_sec':dur,'sample_rate':sr,'channels':ch,'peak':peak,'rms':rms,'qc_status':'pass' if not any(x=='possible_clipping' for x in reasons) else 'flag:'+','.join(reasons), 'qc_note':','.join(reasons)}
 except Exception as e:return {'duration_sec':'','sample_rate':'','channels':'','peak':'','rms':'','qc_status':'error','qc_note':str(e)}
def main():
 audit=json.loads(AUDIT.read_text(encoding='utf-8')); wanted=select(audit); print(f'selected {len(wanted)} candidate rows across {len({r["speakerID"] for r in wanted.values()})} speakers',flush=True); found=[]
 for split,n in SHARDS.items():
  for i in range(n):
   hits=[]
   with fsspec.open(url(split,i),'rb',block_size=1024*1024,cache_type='none').open() as fh:
    for b in pq.ParquetFile(fh).iter_batches(columns=META,batch_size=4096):
     for r in b.to_pylist():
      if (split,r['filename']) in wanted: hits.append(r)
   if not hits: continue
   print(f'reading audio {split} shard {i}: {len(hits)}',flush=True)
   with fsspec.open(url(split,i),'rb',block_size=1024*1024,cache_type='none').open() as fh: table=pq.read_table(fh,columns=META+['audio'])
   names={r['filename'] for r in hits}
   for r in table.to_pylist():
    if r.get('filename') not in names or r.get('province_name')!='NgheAn': continue
    raw=(r.get('audio') or {}).get('bytes');
    if not raw: continue
    p=OUT/split/r['speakerID']/r['filename']; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(raw); found.append({'split':split,'speakerID':r['speakerID'],'filename':r['filename'],'text':r.get('text',''),'gender':r.get('gender',''),'local_path':str(p),**stats(p)})
 found.sort(key=lambda r:(r['split'],r['speakerID'],r['filename'])); fields=['split','speakerID','filename','text','gender','local_path','duration_sec','sample_rate','channels','peak','rms','qc_status','qc_note']
 for name,rs in [('metadata_na_candidates.csv',found)]:
  with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rs)
 pref=[r for r in found if str(r['qc_status']).startswith('pass') and 4<=float(r['duration_sec'])<=15];
 with (ROOT/'metadata_na_preferred_4_15s.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(pref)
 summary={'selected':len(wanted),'extracted':len(found),'speakers':len({r['speakerID'] for r in found}),'preferred_4_15s':len(pref),'by_split':{s:sum(r['split']==s for r in found) for s in SHARDS},'qc_flags':sum(not str(r['qc_status']).startswith('pass') for r in found)}
 (ROOT/'na_candidates_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

