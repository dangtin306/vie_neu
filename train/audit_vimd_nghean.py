from __future__ import annotations
import json
from collections import Counter,defaultdict
from pathlib import Path
import fsspec, pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'vimd_na_audit.json'
SHARDS={'train':103,'valid':13,'test':14}; COLS=['province_name','province_code','filename','text','speakerID','gender']
def url(split,i): return f'https://huggingface.co/datasets/nguyendv02/ViMD_Dataset/resolve/main/data/{split}-{i:05d}-of-{SHARDS[split]:05d}.parquet'
def main():
 rows=[]; all_prov=Counter(); split_counts={}
 for split,n in SHARDS.items():
  scanned=0
  for i in range(n):
   with fsspec.open(url(split,i),'rb',block_size=1024*1024,cache_type='none').open() as fh:
    pf=pq.ParquetFile(fh)
    for b in pf.iter_batches(columns=COLS,batch_size=4096):
     for r in b.to_pylist():
      scanned+=1; all_prov[(r.get('province_name'),r.get('province_code'))]+=1
      if r.get('province_code')==40 or r.get('province_name') in {'NgheAn','Nghệ An'}: r['split']=split; rows.append(r)
   if (i+1)%10==0 or i==n-1: print(f'{split} {i+1}/{n}, scanned={scanned}, NA={len(rows)}',flush=True)
  split_counts[split]=scanned
 by=defaultdict(list)
 for r in rows: by[r['speakerID']].append(r)
 dist=Counter(len(x) for x in by.values())
 result={'source':'nguyendv02/ViMD_Dataset','filter':'province_code == 40 or province_name in {NgheAn,Nghệ An}','audio_downloaded':False,'split_row_counts_scanned':split_counts,'province_values_matching':{str(k):v for k,v in all_prov.items() if k[1]==40 or k[0] in {'NgheAn','Nghệ An'}},'utterances':len(rows),'speakers':len(by),'speaker_distribution':dict(sorted(dist.items())),'speakers_ge_2':sum(v for k,v in dist.items() if k>=2),'speakers_ge_3':sum(v for k,v in dist.items() if k>=3),'speakers_ge_5':sum(v for k,v in dist.items() if k>=5),'speakers_ge_10':sum(v for k,v in dist.items() if k>=10),'speaker_rows':dict(sorted(by.items()))}
 OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in result.items() if k!='speaker_rows'},ensure_ascii=False,indent=2)); print(f'wrote {OUT}')
if __name__=='__main__': main()
