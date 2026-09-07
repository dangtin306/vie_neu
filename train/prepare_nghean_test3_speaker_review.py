from __future__ import annotations
import csv, shutil
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
EXP=ROOT/'output'/'nghean_test3_accent_strong_data'
SOURCE=ROOT/'output'/'nghean_test2_accent_scale'/'nghean_audio_qc.csv'
REVIEW=EXP/'speaker_review'

def main():
    rows=list(csv.DictReader(SOURCE.open(encoding='utf-8-sig',newline='')))
    by=defaultdict(list)
    for r in rows: by[r['speakerID']].append(r)
    speakers={s:sorted(rs,key=lambda x:x['filename']) for s,rs in by.items() if len(rs)>=2}
    if len(speakers)!=77: raise RuntimeError(f'expected 77 eligible speakers, got {len(speakers)}')
    REVIEW.mkdir(parents=True,exist_ok=True); out=[]
    for spk,rs in sorted(speakers.items()):
        chosen=rs[:3]
        folder=REVIEW/spk; folder.mkdir(parents=True,exist_ok=True)
        info=[f'speakerID: {spk}',f'split: {sorted({r["split"] for r in rs})}',f'clips_in_dataset: {len(rs)}','']
        for i,r in enumerate(chosen,1):
            dst=folder/f'clip_{i:02d}.wav'; shutil.copy2(r['local_path'],dst)
            info += [f'clip_{i:02d}: {r["filename"]}',f'duration: {r["duration"]} sec',f'technical_qc: {r["qc_status"]}',f'transcript: {r["transcript"]}','']
        (folder/'info.txt').write_text('\n'.join(info),encoding='utf-8')
        out.append({'speakerID':spk,'split':'/'.join(sorted({r['split'] for r in rs})),'clip_1':chosen[0]['filename'],'clip_2':chosen[1]['filename'] if len(chosen)>1 else '','clip_3':chosen[2]['filename'] if len(chosen)>2 else '','accent_strength':'UNSURE','consistency':'UNSURE','audio_quality':'UNSURE','keep_for_training':'UNSURE','notes':''})
    fields=list(out[0])
    with (EXP/'speaker_accent_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
    (EXP/'README_review.txt').write_text('Nghe it nhat 2 clip cung speaker. Danh gia mau phat am Nghe An, khong can chuan 100%.\n\nSTRONG: mau Nghe An ro o nhieu am/ca cau.\nMEDIUM: co mau Nghe An nhung khong lien tuc.\nWEAK: gan pho thong, chi hoi co mau.\nUNSURE: kho xac dinh.\n\nDien speaker_accent_review.csv. Khong dung waveform/spectrogram/MOSS de tu gan accent_strength.\n',encoding='utf-8')
    print(f'created speaker review: {len(out)} speakers; {sum(len(sorted(rs,key=lambda x:x["filename"])[:3]) for rs in speakers.values())} copied clips; output={EXP}',flush=True)
if __name__=='__main__':main()
