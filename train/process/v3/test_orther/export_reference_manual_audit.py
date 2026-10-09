from __future__ import annotations
import csv, json, shutil
from pathlib import Path
import numpy as np, soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import stft

ROOT=Path(__file__).resolve().parent.parent
SRC=ROOT/'data'/'dataset_haiphong'/'test'
OUT=ROOT/'output'/'test13_scale_feasibility'/'reference_manual_audit'
ITEMS=[('spk_15_0218','15_0294.wav','0218_0294'),('spk_15_0218','15_0295.wav','0218_0295'),('spk_15_0220','15_0301.wav','0220_0301'),('spk_15_0219','15_0299.wav','0219_0299_control')]

def metrics(y,sr):
    y=np.asarray(y,dtype=np.float32); duration=len(y)/sr; peak=float(np.max(np.abs(y))) if len(y) else 0.; rms=float(np.sqrt(np.mean(y*y))) if len(y) else 0.; dc=float(np.mean(y)) if len(y) else 0.; clip=float(np.mean(np.abs(y)>=.999)) if len(y) else 0.
    frame=max(1,int(sr*.02)); e=np.array([np.sqrt(np.mean(y[i:min(len(y),i+frame)]**2)) for i in range(0,len(y),frame)]) if len(y) else np.array([]); th=max(1e-5,float(e.max())*.02) if len(e) else 1e-5; silent=e<=th
    lead=0; trail=0
    if len(silent):
        k=0
        while k<len(silent) and silent[k]: k+=1
        lead=min(len(y),k*frame)/sr; k=len(silent)-1
        while k>=0 and silent[k]: k-=1
        trail=max(0,len(silent)-1-k)*frame/sr
    spans=[]; i=0
    while i<len(silent):
        if not silent[i]: i+=1; continue
        j=i
        while j<len(silent) and silent[j]: j+=1
        spans.append((j-i)*frame/sr); i=j
    internal=[x for x in spans if x<duration and x>.04]
    return {'duration':duration,'sample_rate':sr,'channels':1,'peak':peak,'RMS':rms,'DC_offset':dc,'clipping_ratio':clip,'leading_silence':lead,'trailing_silence':trail,'internal_silence_ratio':sum(internal)/duration if duration else 0.,'longest_internal_silence':max(internal or [0.]),'machine_observation':('peak near full scale' if peak>=.98 else 'no obvious peak clipping; music/noise/reverb not classified by machine')}

def main():
    OUT.mkdir(parents=True,exist_ok=True); plots=[]; checklist=[]
    for spk,fn,label in ITEMS:
        src=SRC/spk/fn
        if not src.exists(): raise FileNotFoundError(src)
        d=OUT/label; d.mkdir(exist_ok=True); shutil.copy2(src,d/'reference.wav')
        y,sr=sf.read(src,always_2d=False); y=np.asarray(y,dtype=np.float32); y=y.mean(axis=1) if y.ndim>1 else y; m=metrics(y,sr); m.update({'speakerID':spk,'filename':fn,'source_path':str(src)})
        (d/'info.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
        t=np.arange(len(y))/sr; fig,ax=plt.subplots(figsize=(14,3)); ax.plot(t,y,lw=.35); ax.set_title(f'{spk} / {fn} waveform'); ax.set_xlabel('Time (s)'); ax.set_ylabel('Amplitude'); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(d/'waveform.png',dpi=150); plt.close(fig)
        f,tt,z=stft(y,fs=sr,nperseg=1024,noverlap=768); db=20*np.log10(np.maximum(np.abs(z),1e-7)); fig,ax=plt.subplots(figsize=(14,4)); im=ax.pcolormesh(tt,f,db,shading='auto',vmin=-100,vmax=-10,cmap='magma'); ax.set_ylim(0,8000); ax.set_title(f'{spk} / {fn} STFT (same scale)'); ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)'); fig.colorbar(im,ax=ax,label='dB'); fig.tight_layout(); fig.savefig(d/'spectrogram.png',dpi=150); plt.close(fig)
        plots.append((label,y,sr))
        checklist.append({'speakerID':spk,'filename':fn,'music_background':'','secondary_speech':'','strong_noise':'','reverb_echo':'','far_microphone':'','volume_abnormal':'','cut_start':'','cut_end':'','long_pause':'','speech_fragmented':'','pronunciation_problem':'','overall_status':'','notes':''})
    fig,axs=plt.subplots(4,2,figsize=(16,14));
    for i,(label,y,sr) in enumerate(plots):
        t=np.arange(len(y))/sr; axs[i,0].plot(t,y,lw=.3); axs[i,0].set_title(label+' waveform'); axs[i,0].set_xlabel('s'); axs[i,0].grid(alpha=.2); f,tt,z=stft(y,fs=sr,nperseg=1024,noverlap=768); db=20*np.log10(np.maximum(np.abs(z),1e-7)); axs[i,1].pcolormesh(tt,f,db,shading='auto',vmin=-100,vmax=-10,cmap='magma'); axs[i,1].set_ylim(0,8000); axs[i,1].set_title(label+' STFT'); axs[i,1].set_xlabel('s')
    fig.tight_layout(); fig.savefig(OUT/'reference_comparison.png',dpi=150); plt.close(fig)
    with (OUT/'listening_checklist.csv').open('w',encoding='utf-8-sig',newline='') as f: w=csv.DictWriter(f,fieldnames=list(checklist[0])); w.writeheader(); w.writerows(checklist)
    print(f'exported {len(ITEMS)} references to {OUT}')
if __name__=='__main__': main()

