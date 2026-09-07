"""TEST13 cross-conditioning diagnostic. Inference and teacher-forced tracing only."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
import numpy as np, torch, soundfile as sf
ROOT=Path(__file__).resolve().parent.parent; SRC=ROOT.parent/'source_code'/'audio_model'/'src'; OUT=ROOT/'output'/'test13_scale_feasibility'; D=OUT/'cross_conditioning_diagnostic'
sys.path.insert(0,str(SRC));sys.path.insert(0,str(ROOT))
import test13_scale_feasibility as t
from overfit_one_sample_test import teacher_forced_logits

MODELS={'Base':None,'TF-LoRA':OUT/'tf_best','Warm98_2':OUT/'warm98_best'}
TEXTS={'error_01':'Hôm nay thời tiết khá dễ chịu.','stable_01':'Cảm ơn bạn, hẹn gặp lại vào ngày mai.'}

def tf_trace(engine,sample):
    hs=t.true_history(engine,sample); rows=[]; losses=[]
    for i,(h,target) in enumerate(zip(hs,sample['codes'])):
        logits=teacher_forced_logits(engine,h,target); ce=[]; ent=[]; top=[]
        for k,z in enumerate(logits):
            p=torch.softmax(z.float(),dim=-1); ce.append(float(torch.nn.functional.cross_entropy(z,target[k].view(1)).cpu())); ent.append(float((-p*torch.log(p.clamp_min(1e-12))).sum().cpu())); top.append(float(p.max().cpu()))
        # EOS head is not part of the audio objective; trace it when available.
        eos=None
        if hasattr(engine.model,'text_lm_head'):
            z=engine.model.text_lm_head(h.to(next(engine.model.text_lm_head.parameters()).dtype)).float(); p=torch.softmax(z,dim=-1); eos=float(p[0,engine.config.speech_generation_end_token_id].cpu())
        rows.append({'frame':i,'mean_ce':float(np.mean(ce)),'codebook_ce':ce,'mean_entropy':float(np.mean(ent)),'mean_top1_prob':float(np.mean(top)),'eos_prob':eos})
    return rows
def main():
    pairs=list(csv.DictReader((ROOT/'test_pairs.csv').open(encoding='utf-8-sig',newline='')))
    r218=next(r for r in pairs if r['speakerID']=='spk_15_0218' and Path(r['reference_path']).stem=='15_0295')
    r219=next(r for r in pairs if r['speakerID']=='spk_15_0219' and Path(r['reference_path']).stem=='15_0299')
    t.PREP_TOTAL=2; ref_engine=t._load_engine(); s218=t.prepare(ref_engine,r218); s219=t.prepare(ref_engine,r219)
    refs={'0218':(s218['speaker_emb'],s218['ref_codes'],s218['codes']),'0219':(s219['speaker_emb'],s219['ref_codes'],s219['codes'])}; del ref_engine; torch.cuda.empty_cache()
    cond={'A':('0218','0218'),'B':('0219','0218'),'C':('0218','0219'),'D':('0219','0219')}; matrix=[]; trace=[]
    for model,path in MODELS.items():
        e=t._load_engine()
        if path:t.load_adapter(e,path)
        for scenario,(ss,rr) in cond.items():
            spk,ref,codes=refs[ss][0],refs[rr][1],refs[ss][2]
            for tid,text in [('error_01',TEXTS['error_01'])]:
                phones=e._resolve_phonemes(None,text); prompt=e._build_prompt_2d(phones,None,np.asarray(ref),e._resolve_style_id()); sample={'prompt':prompt,'codes':codes,'speaker_emb':spk}
                tr=tf_trace(e,sample); loss=float(np.mean([x['mean_ce'] for x in tr]));
                gen,_,stop=t.trace_generation(e,text,(spk,ref),t.SEED); wav=np.asarray(e._decode_codes(gen),dtype=np.float32).reshape(-1); m=t.waveform_metrics(wav,gen); out=D/f'{scenario}_{model.replace("-","")}_{tid}.wav'; D.mkdir(parents=True,exist_ok=True); sf.write(out,wav,48000)
                matrix.append({'scenario':scenario,'speaker_embedding_source':ss,'reference_codes_source':rr,'text_id':tid,'model':model,'teacher_forced_ce':loss,'entropy':float(np.mean([x['mean_entropy'] for x in tr])),'top1_prob':float(np.mean([x['mean_top1_prob'] for x in tr])),'eos_prob':tr[-1]['eos_prob'],'duration':len(wav)/48000.,'frames':len(gen),'eos':stop=='eos','hit_max':stop=='max_new_frames','silence_ratio':m['silence_ratio'],'longest_silence':m['longest_silence_sec'],'wav_path':str(out)})
                for x in tr: trace.append({'scenario':scenario,'speaker_embedding_source':ss,'reference_codes_source':rr,'text_id':tid,'model':model,**x})
        # stable text under native conditions A and D
        for scenario,(ss,rr) in {'A':cond['A'],'D':cond['D']}.items():
            spk,ref,codes=refs[ss][0],refs[rr][1],refs[ss][2]; text=TEXTS['stable_01']; prompt=e._build_prompt_2d(e._resolve_phonemes(None,text),None,np.asarray(ref),e._resolve_style_id()); sample={'prompt':prompt,'codes':codes,'speaker_emb':spk}
            for x in tf_trace(e,sample): trace.append({'scenario':scenario,'speaker_embedding_source':ss,'reference_codes_source':rr,'text_id':'stable_01','model':model,**x})
        del e; torch.cuda.empty_cache()
    fields=list(matrix[0]);
    with (OUT/'test13_cross_conditioning_matrix.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(matrix)
    tf_fields=list(trace[0]);
    with (OUT/'test13_same_history_trace_v2.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=tf_fields);w.writeheader();w.writerows(trace)
    summary={'conditions':{'A':'S0218+R0218','B':'S0219+R0218','C':'S0218+R0219','D':'S0219+R0219'},'models':list(MODELS),'sampling':'TEST13 .8/top_k25/top_p.95/repetition_penalty1.2/max300','matrix_rows':len(matrix),'trace_rows':len(trace),'speaker_vectors':{k:{'speaker_embedding_norm':float(np.linalg.norm(v[0]))} for k,v in refs.items()},'interpretation':'Automatic evidence only; no accent conclusion.'}
    (OUT/'test13_cross_conditioning_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); print(f'wrote outputs to {OUT}')
if __name__=='__main__':main()
