#!/usr/bin/env python3
"""Clone the Nghệ An reference WAV with VieNeu v3 Turbo."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HF_HOME = Path("/home/mediatech/dangtin/trainning/application/huggingface")
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME / "hub"))

from vieneu import Vieneu

REFERENCE = ROOT / "/home/mediatech/dangtin/trainning/ai/vie_neu/train/process/v3/test_6/output/nghean_region_no_anchor/dataset/raw_audio/0008_37_0006.wav"
OUTPUT = Path(__file__).resolve().parent / "output.wav"
TEXT = "Không chỉ nhà trong hẻm, những căn mặt tiền có giá trị lớn cũng bắt đầu xuất hiện mức điều chỉnh đáng kể. Anh Đình Toàn, môi giới nhà phố, cho biết đang có rổ hàng hàng chục căn tại Phú Nhuận, quận 10, quận 5 và quận 3 cũ với giá chào thấp hơn đáng kể so với vài tháng trước. Một căn mặt tiền đường Lý Thường Kiệt, rộng 56 m2, hiện được ký gửi với giá 16,7 tỷ đồng, giảm hơn 5 tỷ đồng so với mức chủ nhà từng kỳ vọng hồi tháng 5. Chủ muốn bán nhanh để tất toán khoản vay đến hạn."


def main():
    if not REFERENCE.is_file():
        raise FileNotFoundError(REFERENCE)

    print(f"reference: {REFERENCE}", flush=True)
    print(f"text: {TEXT}", flush=True)
    tts = Vieneu(mode="v3turbo")
    audio = tts.infer(TEXT, ref_audio=str(REFERENCE), denoise=True)
    tts.save(audio, OUTPUT)
    print(f"saved: {OUTPUT} ({len(audio) / tts.sample_rate:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
