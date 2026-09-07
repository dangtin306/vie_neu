# -*- coding: utf-8 -*-
"""VieNeu-TTS test server with a disk checkpoint cache and per-request cleanup."""

import os
import gc
import shutil
import subprocess
import sys
import time
import urllib.request
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock, Thread

NL = chr(10)

# The Windows console may default to cp1258; VieNeu logs contain Vietnamese text.
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Keep the worker protocol pipe separate from diagnostic/model-loading logs.
_WORKER_PROTOCOL_STDOUT = sys.stdout


def get_max_cpu_threads() -> int:
    """Tự động đo lường và phát hiện số luồng CPU tối đa của máy đang chạy (tương thích đa máy chủ)."""
    try:
        count = os.cpu_count()
        if count and count > 0:
            return count
    except Exception:
        pass
    try:
        count = multiprocessing.cpu_count()
        if count and count > 0:
            return count
    except Exception:
        pass
    return 4


# === Cấu hình TỐI ĐA HÓA TẤT CẢ CÁC LUỒNG CPU ĐỘNG THEO PHẦN CỨNG MÁY CHỦ ===
CPU_THREADS = get_max_cpu_threads()
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(CPU_THREADS)
os.environ["RAYON_NUM_THREADS"] = str(CPU_THREADS)  # Tối đa hóa đa luồng Rust cho sea-g2p
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["MKL_DYNAMIC"] = "FALSE"

# Set caches before importing torch, huggingface_hub, or VieNeu modules.
CACHE_DIR = Path('F:/ai/cache/cuda/vie_neu')
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TORCH_KERNEL_CACHE_DIR = Path('F:/ai/cache/torch_kernels')
TORCH_KERNEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / 'huggingface' / 'hub')
os.environ["TORCH_HOME"] = str(CACHE_DIR / 'torch')

import numpy as np
import soundfile as sf
import torch
from flask import Flask, jsonify, request, send_file
from pydub import AudioSegment

# Áp dụng cấu hình đa luồng tối đa cho PyTorch CPU theo phần cứng thực tế
torch.set_num_threads(CPU_THREADS)
try:
    torch.set_num_interop_threads(min(8, CPU_THREADS))
except RuntimeError:
    pass

# === Cấu hình TỐI ĐA HÓA HIỆU NĂNG GPU NVIDIA RTX (CUDA / Tensor Cores / cuDNN) ===
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_DIR / 'source_code'
AUDIO_MODEL_SRC = SOURCE_ROOT / 'audio_model' / 'src'
if AUDIO_MODEL_SRC.exists():
    sys.path.insert(0, str(AUDIO_MODEL_SRC))

from vieneu import Vieneu
from vieneu_utils.phonemize_text import phonemize_text_with_emotions

app = Flask(__name__)
inference_lock = Lock()

OUTPUT_DIR = Path('D:/hustmedia/python/tts/output')
OUTPUT_WAV = OUTPUT_DIR / 'output_vieneu_persistent.wav'
OUTPUT_MP3 = OUTPUT_DIR / 'output_vieneu_persistent.mp3'
FFMPEG_EXE = Path('D:/hustmedia/application/ffmpeg/bin/ffmpeg.exe')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
_engine = None
_engine_lock = Lock()


def get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = Vieneu(
                mode='v3turbo',
                device=device,
                backend='auto',
                threads=CPU_THREADS,
                max_batch_size=8,
            )
            # Pre-warm sea-g2p (Rust CPU) and MOSS Codec (GPU) and Batch Engine so they stay pinned in RAM/VRAM
            try:
                _ = phonemize_text_with_emotions('khởi động')
            except Exception:
                pass
        return _engine


def synthesize(text: str) -> Path:
    engine = get_engine()
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            # Gộp tối đa 8 đoạn vào GPU cùng 1 lúc (Static Batching)
            audio = engine.infer(text=text, batch_size=8, max_chars=110)
        engine.save(audio, OUTPUT_WAV)

        if FFMPEG_EXE.exists():
            cmd = [
                str(FFMPEG_EXE),
                '-y',
                '-threads', str(min(16, CPU_THREADS)),
                '-i', str(OUTPUT_WAV),
                '-codec:a', 'libmp3lame',
                '-b:a', '192k',
                str(OUTPUT_MP3),
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return OUTPUT_MP3
        else:
            AudioSegment.from_wav(OUTPUT_WAV).export(OUTPUT_MP3, format='mp3', bitrate='192k')
            return OUTPUT_MP3
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


worker_mode = '--worker' in sys.argv or '--persistent-worker' in sys.argv
if worker_mode:
    sys.stdout = sys.stderr
    print(f'[VieNeu Worker] Khởi động trên {device} (Phát hiện tự động CPU: {CPU_THREADS} threads & Max GPU Boost), cache: {CACHE_DIR}', file=sys.stderr)
    try:
        get_engine()
        print(f'[VieNeu Worker] Mô hình VieNeu-TTS v3, sea-g2p và MOSS Codec đã thường trú sẵn trong RAM & VRAM!', file=sys.stderr)
        _WORKER_PROTOCOL_STDOUT.write(json.dumps({'ready': True, 'device': device}) + NL)
        _WORKER_PROTOCOL_STDOUT.flush()
    except Exception as exc:
        _WORKER_PROTOCOL_STDOUT.write(json.dumps({'ready': False, 'message': str(exc)}) + NL)
        _WORKER_PROTOCOL_STDOUT.flush()
        sys.exit(1)

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            req = json.loads(line)
            if req.get('stop'):
                break
            text = req.get('text', '').strip()
            if not text:
                _WORKER_PROTOCOL_STDOUT.write(json.dumps({'ok': False, 'message': 'Missing or empty "text"'}) + NL)
                _WORKER_PROTOCOL_STDOUT.flush()
                continue

            out_path = synthesize(text)
            _WORKER_PROTOCOL_STDOUT.write(json.dumps({
                'ok': True,
                'output_path': str(out_path),
                'wav_path': str(OUTPUT_WAV),
            }) + NL)
            _WORKER_PROTOCOL_STDOUT.flush()
        except Exception as exc:
            _WORKER_PROTOCOL_STDOUT.write(json.dumps({'ok': False, 'message': str(exc)}) + NL)
            _WORKER_PROTOCOL_STDOUT.flush()

    sys.exit(0)


@app.get('/health')
def health():
    return jsonify({'status': 'ok', 'device': device, 'cache': str(CACHE_DIR), 'threads': CPU_THREADS})


@app.post('/tts')
def tts():
    data = request.get_json(force=True) or {}
    text = str(data.get('text', '')).strip()
    text = ''.join(char for char in text if not 0xD800 <= ord(char) <= 0xDFFF)
    if not text:
        return jsonify({'error': 'Missing or empty "text" field'}), 400

    with inference_lock:
        try:
            output = synthesize(text)
        except Exception as exc:
            return jsonify({'status': 'error', 'message': str(exc)}), 500
    return send_file(output, mimetype='audio/mpeg', as_attachment=False)


if __name__ == '__main__':
    print(f'VieNeu-TTS standalone test server running on {device}')
    app.run(host='0.0.0.0', port=8791)
