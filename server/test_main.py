# -*- coding: utf-8 -*-
"""Pipeline Tổng Thể: Nhúng và điều phối 3 bước qua test_text, test_audio và test_output."""

import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.stdout is not None and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr is not None and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Thêm thư mục hiện tại vào sys.path để import các file con
SERVER_DIR = Path(__file__).resolve().parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

# Nhúng trực tiếp 3 module con:
from test_text import preprocess_text
from test_audio import get_audio_engine, synthesize_speech
from test_output import export_audio, print_audio_stats


def run_pipeline(
    raw_text: str,
    output_filename: str = 'vieneu_pipeline.wav',
    voice: str = None,
    ref_audio: str = None,
):
    """Hàm điều phối toàn bộ luồng xử lý 3 bước chuẩn."""
    print('=' * 75)
    print('  QUY TRÌNH TTS 3 BƯỚC: TEST_TEXT ➔ TEST_AUDIO ➔ TEST_OUTPUT')
    print('=' * 75)

    start_total = time.time()
    output_dir = SERVER_DIR / 'outputs'
    output_path = output_dir / output_filename

    # -------------------------------------------------------------
    # BƯỚC 1: Tiền xử lý văn bản (Gọi từ test_text.py)
    # -------------------------------------------------------------
    print('')
    print('>>> [BƯỚC 1] TIỀN XỬ LÝ VĂN BẢN (GỌI TỪ test_text.py)')
    t0 = time.time()
    text_info = preprocess_text(raw_text)
    t_text = time.time() - t0

    print(f'  - Văn bản thô : {text_info["raw"]}')
    print(f'  - Đã chuẩn hóa: {text_info["normalized"]}')
    print(f'  - Chuỗi âm vị : {text_info["phonemes_tts"]}')
    print(f'  - Thời gian   : {t_text:.4f} giây')

    # -------------------------------------------------------------
    # BƯỚC 2: Mô hình AI tổng hợp âm thanh (Gọi từ test_audio.py)
    # -------------------------------------------------------------
    print('')
    print('>>> [BƯỚC 2] MÔ HÌNH AI TỔNG HỢP ÂM THANH (GỌI TỪ test_audio.py)')
    t0 = time.time()
    engine = get_audio_engine()
    audio_data = synthesize_speech(
        engine,
        text=text_info['raw'],
        ref_audio=ref_audio,
        voice=voice,
    )
    t_audio = time.time() - t0
    print(f'  - Đã sinh xong Speech Tokens & giải mã Codec trong {t_audio:.2f} giây')

    # -------------------------------------------------------------
    # BƯỚC 3: Xuất và kiểm tra định dạng file (Gọi từ test_output.py)
    # -------------------------------------------------------------
    print('')
    print('>>> [BƯỚC 3] XUẤT VÀ KIỂM TRA FILE AUDIO (GỌI TỪ test_output.py)')
    t0 = time.time()
    stats = export_audio(audio_data, output_path, engine=engine)
    t_output = time.time() - t0
    print_audio_stats(stats, title='KẾT QUẢ FILE AUDIO ĐÃ LƯU')
    print(f'  - Thời gian ghi file : {t_output:.4f} giây')

    total_time = time.time() - start_total
    print('')
    print('=' * 75)
    print(f'  HOÀN THÀNH TOÀN BỘ QUY TRÌNH TRONG: {total_time:.2f} GIÂY!')
    print('=' * 75)
    return stats


def main():
    sample_text = (
        'Xin chào các bạn! Hôm nay ngày 19/08/2026, '
        'chúng ta đã tách và kết nối thành công 3 bước trong quy trình TTS [cười]. '
        'Hệ thống AI xử lý siêu nhanh và chuẩn xác.'
    )
    run_pipeline(sample_text, output_filename='vieneu_main_3steps.wav')


if __name__ == '__main__':
    main()
