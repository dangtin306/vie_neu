# -*- coding: utf-8 -*-
"""Bước 3: Xuất và kiểm tra định dạng file âm thanh đầu ra (test_output)."""

import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any

# Ensure UTF-8 output on Windows console
if sys.stdout is not None and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr is not None and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import soundfile as sf
import numpy as np


def export_audio(
    audio_data: Any,
    output_path: Union[str, Path],
    sample_rate: int = 48000,
    engine: Optional[Any] = None,
) -> Dict[str, Any]:
    """Hàm chuyên trách xuất dữ liệu âm thanh ra file WAV và trả về thông số kỹ thuật."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Lưu file qua engine của Vieneu nếu có, hoặc dùng soundfile
    if engine is not None and hasattr(engine, 'save'):
        engine.save(audio_data, out_file)
    else:
        # Nếu audio_data là numpy array hoặc torch tensor
        if hasattr(audio_data, 'detach'):
            audio_data = audio_data.detach().cpu().numpy()
        if isinstance(audio_data, np.ndarray):
            sf.write(str(out_file), audio_data, sample_rate)
        else:
            raise ValueError('Không thể nhận diện định dạng audio_data để xuất file.')

    # 2. Đọc lại và kiểm tra thông số kỹ thuật
    return inspect_audio_file(out_file)


def inspect_audio_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Đọc và phân tích thông số kỹ thuật của file âm thanh."""
    fpath = Path(file_path)
    if not fpath.exists():
        raise FileNotFoundError(f'Không tìm thấy file: {fpath}')

    file_size_bytes = fpath.stat().st_size
    info = sf.info(str(fpath))

    stats = {
        'file_path': str(fpath.resolve()),
        'file_name': fpath.name,
        'file_size_bytes': file_size_bytes,
        'file_size_kb': round(file_size_bytes / 1024, 2),
        'duration_seconds': round(info.duration, 2),
        'sample_rate': info.samplerate,
        'channels': info.channels,
        'format': info.format,
        'subtype': info.subtype,
    }
    return stats


def print_audio_stats(stats: Dict[str, Any], title: str = 'THÔNG SỐ FILE ÂM THANH XUẤT RA'):
    """In bảng thông số file âm thanh đẹp mắt ra console."""
    print('')
    print(f'[{title}]')
    print(f'  - Tên file     : {stats["file_name"]}')
    print(f'  - Đường dẫn    : {stats["file_path"]}')
    print(f'  - Dung lượng   : {stats["file_size_kb"]} KB ({stats["file_size_bytes"]:,} bytes)')
    print(f'  - Thời lượng   : {stats["duration_seconds"]} giây')
    print(f'  - Sample Rate  : {stats["sample_rate"]} Hz')
    print(f'  - Định dạng    : {stats["format"]} ({stats["subtype"]})')
    print(f'  - Số kênh      : {stats["channels"]} (Mono)' if stats["channels"] == 1 else f'  - Số kênh      : {stats["channels"]} (Stereo)')


def main():
    print('=' * 70)
    print('  [BƯỚC 3] TEST XUẤT VÀ KIỂM TRA FILE ÂM THANH (TEST_OUTPUT)')
    print('=' * 70)

    output_dir = Path(__file__).resolve().parent / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_file = output_dir / 'test_output_tone.wav'

    # Tạo một file âm thanh mẫu (1 giây sine wave 440Hz 48kHz) để test độc lập
    sample_rate = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)

    print('[test_output] Đang ghi file âm thanh mẫu kiểm thử...')
    stats = export_audio(sine_wave, sample_file, sample_rate=sample_rate)
    print_audio_stats(stats, title='KẾT QUẢ XUẤT FILE MẪU')

    print('')
    print('=' * 70)
    print('  HOÀN THÀNH TEST_OUTPUT THÀNH CÔNG!')
    print('=' * 70)


if __name__ == '__main__':
    main()
