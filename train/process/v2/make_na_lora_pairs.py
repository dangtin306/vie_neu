from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
rows=list(csv.DictReader((ROOT/'metadata_na_candidates.csv').open(encoding='utf-8-sig',newline='')))
by=defaultdict(list)
for r in rows:
    if r['qc_status'].startswith('pass'): by[r['speakerID']].append(r)
pairs=[]
for spk,rs in sorted(by.items()):
    if len(rs)<2: continue
    rs.sort(key=lambda r:r['filename'])
    a,b=rs[0],rs[1]
    pairs.append({'speakerID':spk,'reference_path':a['local_path'],'target_path':b['local_path'],'target_text':b['text'],'reference_duration':a['duration_sec'],'target_duration':b['duration_sec'],'split':'train'})
train_speakers={r['speakerID'] for r in pairs[:9]}; valid_speakers={r['speakerID'] for r in pairs[9:]}
for r in pairs: r['split']='train' if r['speakerID'] in train_speakers else 'valid'
fields=['speakerID','reference_path','target_path','target_text','reference_duration','target_duration','split']
for name,subset in [('na_train_pairs.csv',[r for r in pairs if r['split']=='train']),('na_valid_pairs.csv',[r for r in pairs if r['split']=='valid']),('na_test_pairs.csv',[r for r in pairs if r['split']=='valid'])]:
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(subset)
print(f'train pairs={len([r for r in pairs if r["split"]=="train"])} valid/demo pairs={len(valid_speakers)} speakers={len(pairs)}')
