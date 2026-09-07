# -*- coding: utf-8 -*-
"""Bước 2: Mô hình AI VieNeu-TTS tổng hợp âm thanh (audio_model)."""

import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.stdout is not None and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr is not None and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Cache configuration
CACHE_DIR = Path(os.environ.get('VIE_NEU_CACHE_DIR', Path(__file__).resolve().parents[1] / 'cache' / 'cuda' / 'vie_neu'))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(CACHE_DIR)
os.environ['HUGGINGFACE_HUB_CACHE'] = str(CACHE_DIR / 'huggingface' / 'hub')

SOURCE_ROOT = Path(__file__).resolve().parents[1] / 'source_code' / 'audio_model'
sys.path.insert(0, str(SOURCE_ROOT / 'src'))
sys.path.insert(0, str(SOURCE_ROOT))

from vieneu import Vieneu

_engine_instance = None


def get_audio_engine(mode='v3turbo', device='auto', backend='auto'):
    """Khởi tạo hoặc lấy instance singleton của engine VieNeu-TTS."""
    global _engine_instance
    if _engine_instance is None:
        print(f'[test_audio] Đang nạp mô hình VieNeu-TTS ({mode}, device={device})...')
        _engine_instance = Vieneu(
            mode=mode,
            device=device,
            backend=backend,
            threads=os.cpu_count() or 1,
        )
        print(f'[test_audio] Backend đang sử dụng: {getattr(_engine_instance, "backend", "auto")}')
    return _engine_instance


def synthesize_speech(engine, text: str, ref_audio=None, voice=None):
    """Hàm tổng hợp âm thanh từ văn bản/âm vị qua mô hình AI."""
    if ref_audio:
        return engine.infer(text=text, ref_audio=ref_audio)
    elif voice:
        return engine.infer(text=text, voice=voice)
    else:
        return engine.infer(text=text)


def main():
    print('=' * 70)
    print('  [BƯỚC 2] TEST MÔ HÌNH AI SINH ÂM THANH (TEST_AUDIO)')
    print('=' * 70)

    output_dir = Path(__file__).resolve().parent / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'vieneu_test.wav'

    text = 'Xin chào, đây là bài kiểm tra độc lập của test_audio.'
    engine = get_audio_engine()
    print(f'[test_audio] Đang tổng hợp câu: "{text}"')
    audio = synthesize_speech(engine, text)

    engine.save(audio, output_path)
    print(f'[test_audio] Đã lưu file tại: {output_path}')
    print('=' * 70)


if __name__ == '__main__':
    main()
