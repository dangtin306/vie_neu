from __future__ import annotations
import csv, json, shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'output'/'test13_scale_feasibility'
RAW=OUT/'blind_audio'
RESULT=OUT/'blind' if (OUT/'blind'/'summary.json').exists() else OUT
PAIRS=list(csv.DictReader((ROOT/'test_pairs.csv').open(encoding='utf-8-sig',newline='')))
SENTS=[('target_01','Hôm nay thời tiết khá dễ chịu.'),('target_02','Chiều nay chúng ta sẽ gặp nhau ở đâu?'),('target_03','Tôi vừa hoàn thành công việc và đang chuẩn bị về nhà.')]
MODELS={'base':'base','tf':'tf','warm':'warm98_2'}

def wav(model,i,j):
    hits=sorted((RAW/model).glob(f'case_{i:02d}_{j:02d}_*.wav'))
    if not hits: raise FileNotFoundError(f'{model} case {i}/{j}')
    return hits[-1]

def metadata(i,j):
    r=PAIRS[i]; sid,text=SENTS[j]
    sm=json.loads((RESULT/'summary.json').read_text(encoding='utf-8'))
    data={}
    for key,model in MODELS.items():
        v=sm[model]['free_run']['cases'][r['speakerID']][f"pair_{i:02d}_{Path(r['target_path']).stem}"][sid]
        data[key]={'duration_sec':v['duration_sec'],'EOS':v['eos'],'hit_max':v['hit_max'],'silence_ratio':v['silence_ratio'],'longest_silence':v['longest_silence_sec']}
    return {'speakerID':r['speakerID'],'pair_id':f"pair_{i:02d}_{Path(r['target_path']).stem}",'target_id':Path(r['target_path']).stem,'text':text,'sentence_id':sid,'reference_path':r['reference_path'],'mapping_original':{'base':str(wav('base',i,j)),'tf':str(wav('tf',i,j)),'warm98_2':str(wav('warm98_2',i,j))},'metrics':data}

def copy_case(dest,i,j,info_name=None):
    dest.mkdir(parents=True,exist_ok=True); info=metadata(i,j)
    shutil.copy2(PAIRS[i]['reference_path'],dest/'case05_reference.wav' if dest.name=='manual_compare_case05' else 'reference.wav')
    for key,model in MODELS.items():
        name={'base':'case05_base.wav','tf':'case05_tf.wav','warm':'case05_warm.wav'}.get(key, f'{key}.wav')
        shutil.copy2(wav(model,i,j),dest/name)
    (dest/(info_name or 'info.json')).write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    case05=OUT/'manual_compare_case05'; copy_case(case05,5,0,'case05_info.json')
    # Six additional labelled diagnostics: two normal, two silence-heavy, one long, one short.
    chosen=[('normal_01',4,1),('normal_02',7,1),('silence_high_01',10,0),('silence_high_02',6,0),('duration_long',9,0),('duration_short',3,1)]
    root=OUT/'diagnostic_6'; root.mkdir(parents=True,exist_ok=True)
    for label,i,j in chosen:
        d=root/label; d.mkdir(exist_ok=True)
        info=metadata(i,j); info['diagnostic_label']=label
        shutil.copy2(PAIRS[i]['reference_path'],d/'reference.wav')
        for key,model in MODELS.items(): shutil.copy2(wav(model,i,j),d/f'{key}.wav')
        (d/'info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'case05: {case05}')
    print(f'diagnostic: {root}')
if __name__=='__main__': main()
