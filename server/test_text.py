# -*- coding: utf-8 -*-
"""Bước 1: Tiền xử lý văn bản và chuyển đổi âm vị (sea-g2p)."""

import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.stdout is not None and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr is not None and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add audio_model src to sys.path for vieneu_utils
SOURCE_ROOT = Path(__file__).resolve().parents[1] / 'source_code'
AUDIO_MODEL_DIR = SOURCE_ROOT / 'audio_model' / 'src'
if AUDIO_MODEL_DIR.exists():
    sys.path.insert(0, str(AUDIO_MODEL_DIR))

from sea_g2p import SEAPipeline, Normalizer, G2P, punc_norm
from vieneu_utils.phonemize_text import phonemize_text

# Khởi tạo singleton processor
_normalizer = None
_pipeline = None


def get_text_processors():
    global _normalizer, _pipeline
    if _normalizer is None:
        _normalizer = Normalizer(lang='vi')
    if _pipeline is None:
        _pipeline = SEAPipeline(lang='vi')
    return _normalizer, _pipeline


def preprocess_text(raw_text: str) -> dict:
    """Hàm xử lý văn bản thô -> chuẩn hóa -> chuỗi âm vị (phonemes)."""
    normalizer, pipeline = get_text_processors()
    normalized = normalizer.normalize(raw_text)
    phonemes_sea = pipeline.run(raw_text)
    phonemes_tts = phonemize_text(raw_text)
    return {
        'raw': raw_text,
        'normalized': normalized,
        'phonemes_sea': phonemes_sea,
        'phonemes_tts': phonemes_tts,
    }


def main():
    print('=' * 70)
    print('  [BƯỚC 1] TEST TIỀN XỬ LÝ VĂN BẢN VÀ CHUYỂN ĐỔI ÂM VỊ (TEST_TEXT)')
    print('=' * 70)

    test_samples = [
        'Xin chào! Đây là bài kiểm tra xử lý văn bản tiếng Việt.',
        'Hôm nay 19/08/2026, tôi vừa mua 2.5kg táo giá 150k tại Q.1, TP.HCM lúc 14:30.',
        'Mô hình AI machine learning này chạy real-time rất cool và smooth.',
        'Thật là tuyệt vời [cười], dự án này đã hoạt động mượt mà rồi [thở dài].',
    ]

    for idx, text in enumerate(test_samples, 1):
        print('')
        print(f'--- Mẫu {idx} ---')
        res = preprocess_text(text)
        print(f'[Văn bản gốc]   : {res["raw"]}')
        print(f'[Sau chuẩn hóa] : {res["normalized"]}')
        print(f'[Phonemes G2P]  : {res["phonemes_sea"]}')
        print(f'[Phonemes TTS]  : {res["phonemes_tts"]}')

    print('')
    print('=' * 70)
    print('  HOÀN THÀNH TEST_TEXT THÀNH CÔNG!')
    print('=' * 70)


if __name__ == '__main__':
    main()
