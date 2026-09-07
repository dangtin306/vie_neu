from __future__ import annotations
import csv,json,math,wave,sys,os
from collections import defaultdict
from pathlib import Path
import fsspec, pyarrow.parquet as pq
import numpy as np

ROOT=Path(__file__).resolve().parent; AUDIT=ROOT/'vimd_na_audit.json'; OUT=ROOT/'output'/'nghean_test2_accent_scale'/'audio'; SHARDS={'train':103,'valid':13,'test':14}; META=['province_name','province_code','filename','text','speakerID','gender']
LOG=OUT.parent/'extract.log'

class Tee:
 def __init__(self,*streams): self.streams=streams
 def write(self,data):
  for s in self.streams: s.write(data); s.flush()
 def flush(self):
  for s in self.streams: s.flush()
def url(s,i): return f'https://huggingface.co/datasets/nguyendv02/ViMD_Dataset/resolve/main/data/{s}-{i:05d}-of-{SHARDS[s]:05d}.parquet'
def wavstats(p):
 try:
  with wave.open(str(p),'rb') as w:
   raw=w.readframes(w.getnframes()); a=np.frombuffer(raw,dtype='<i2').astype(np.float32)/32768.; ch=w.getnchannels(); sr=w.getframerate(); a=a.reshape(-1,ch) if ch>1 else a
   return {'duration_sec':float(len(a)/sr),'sample_rate':sr,'channels':ch,'peak':float(np.max(np.abs(a))) if len(a) else 0.,'rms':float(math.sqrt(np.mean(a*a))) if len(a) else 0.,'qc_status':'pass' if len(a) else 'error:empty'}
 except Exception as e:return {'duration_sec':'','sample_rate':'','channels':'','peak':'','rms':'','qc_status':'error:'+str(e)}
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 log_handle=LOG.open('a',encoding='utf-8',buffering=1)
 sys.stdout=Tee(sys.__stdout__,log_handle); sys.stderr=Tee(sys.__stderr__,log_handle)
 print(f'START extract_nghean_test2 pid={os.getpid()}',flush=True)
 audit=json.loads(AUDIT.read_text(encoding='utf-8')); eligible={}
 for spk,rs in audit['speaker_rows'].items():
  if len(rs)>=2:
   for r in rs: eligible[(r['split'],r['filename'])]=r
 found=defaultdict(list)
 for split,n in SHARDS.items():
  for i in range(n):
   print(f'scan metadata {split} shard {i+1}/{n}',flush=True)
   with fsspec.open(url(split,i),'rb',block_size=1024*1024,cache_type='none').open() as fh:
    for b in pq.ParquetFile(fh).iter_batches(columns=META,batch_size=4096):
     for r in b.to_pylist():
      k=(split,r['filename'])
      if k in eligible and r.get('province_name')=='NgheAn': found[(split,i)].append(r)
   if found.get((split,i)): print(f'metadata {split} shard {i}: {len(found[(split,i)])}',flush=True)
  print(f'finished metadata scan {split}: {sum(len(found.get((split,j),[])) for j in range(n))} eligible rows',flush=True)
 rows=[]
 for (split,i),wanted in found.items():
  names={r['filename'] for r in wanted}; existing=[]
  for r in wanted: existing.append(OUT/split/r['speakerID']/r['filename'])
  if not all(p.is_file() for p in existing):
   print(f'reading audio {split} shard {i}: {len(names)}',flush=True)
   with fsspec.open(url(split,i),'rb',block_size=1024*1024,cache_type='none').open() as fh: table=pq.read_table(fh,columns=META+['audio'])
   for r in table.to_pylist():
    if r.get('filename') not in names or r.get('province_name')!='NgheAn': continue
    raw=(r.get('audio') or {}).get('bytes')
    if not raw: print('skip no audio',split,r['filename']); continue
    p=OUT/split/r['speakerID']/r['filename']; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(raw)
  for r,p in zip(wanted,existing):
   if p.is_file(): rows.append({'split':split,'speakerID':r['speakerID'],'filename':r['filename'],'text':r.get('text',''),'local_path':str(p),**wavstats(p)})
 rows.sort(key=lambda x:(x['split'],x['speakerID'],x['filename'])); fields=['split','speakerID','filename','text','local_path','duration_sec','sample_rate','channels','peak','rms','qc_status']
 with (ROOT/'output'/'nghean_test2_accent_scale'/'metadata_na_extracted.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 pairs=[]
 for (split,spk),rs in sorted({k:list(v) for k,v in defaultdict(list).items()}.items()): pass
 by=defaultdict(list)
 for r in rows:
  if str(r['qc_status']).startswith('pass'): by[(r['split'],r['speakerID'])].append(r)
 for (split,spk),rs in sorted(by.items()):
  rs.sort(key=lambda x:x['filename']);
  for i in range(max(0,len(rs)-1)):
   a,b=rs[i],rs[i+1]; pairs.append({'speakerID':spk,'reference_path':a['local_path'],'target_path':b['local_path'],'target_text':b['text'],'reference_duration':a['duration_sec'],'target_duration':b['duration_sec'],'split':split})
 pf=['speakerID','reference_path','target_path','target_text','reference_duration','target_duration','split']
 for s in SHARDS:
  with (ROOT/'output'/'nghean_test2_accent_scale'/f'na_{s}_pairs.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=pf);w.writeheader();w.writerows([x for x in pairs if x['split']==s])
 summ={'audit_utterances':audit['utterances'],'audit_speakers':audit['speakers'],'eligible_speakers':audit['speakers_ge_2'],'extracted':len(rows),'by_split':{s:{'clips':sum(r['split']==s for r in rows),'speakers':len({r['speakerID'] for r in rows if r['split']==s})} for s in SHARDS},'pairs':{s:sum(x['split']==s for x in pairs) for s in SHARDS}}
 (ROOT/'output'/'nghean_test2_accent_scale'/'dataset_audit.json').write_text(json.dumps(summ,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(summ,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
