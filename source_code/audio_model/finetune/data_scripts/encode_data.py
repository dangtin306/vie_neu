"""Encode VieNeu metadata with parallel CPU-side audio loading."""

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import librosa
import torch
from neucodec import NeuCodec
from tqdm import tqdm


def _load_audio(item):
    filename, text, audio_path = item
    try:
        wav, _ = librosa.load(audio_path, sr=16000, mono=True)
        return filename, text, wav, None
    except Exception as exc:
        return filename, text, None, exc


def encode_dataset(dataset_dir="finetune/dataset", max_samples=2000):
    metadata_path = os.path.join(dataset_dir, "metadata_cleaned.csv")
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(dataset_dir, "metadata.csv")

    output_path = os.path.join(dataset_dir, "metadata_encoded.csv")
    raw_audio_dir = os.path.join(dataset_dir, "raw_audio")
    if not os.path.exists(metadata_path):
        print("Không tìm thấy metadata.csv hoặc metadata_cleaned.csv")
        return

    print("🦜 Đang tải NeuCodec model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(device)
    codec.eval()

    with open(metadata_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    random.shuffle(lines)
    if len(lines) > max_samples:
        lines = lines[:max_samples]

    items = []
    skipped_count = 0
    for line in lines:
        parts = line.strip().split("|")
        if len(parts) < 2:
            continue
        filename, text = parts[0], parts[1]
        audio_path = os.path.join(raw_audio_dir, filename)
        if not os.path.exists(audio_path):
            skipped_count += 1
            continue
        items.append((filename, text, audio_path))

    # CPU workers decode/normalize WAV files concurrently.  NeuCodec remains
    # a single model on the selected device to avoid duplicating GPU memory.
    workers = max(
        1, int(os.environ.get("VIENEU_ENCODE_WORKERS", os.cpu_count() or 1))
    )
    print(f"🦜 Encode {len(items)} mẫu với {workers} CPU workers...")
    lines_to_write = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        loaded = pool.map(_load_audio, items)
        for filename, text, wav, load_error in tqdm(
            loaded, total=len(items), desc="encode"
        ):
            if load_error is not None:
                print(f"🦜 Lỗi đọc file {filename}: {load_error}")
                skipped_count += 1
                continue
            try:
                wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    codes = codec.encode_code(wav_tensor)
                codes = codes.squeeze(0).squeeze(0).cpu().numpy().flatten().tolist()
                codes = [int(x) for x in codes]
                if not codes or not all(0 <= c < 65536 for c in codes):
                    skipped_count += 1
                    continue
                lines_to_write.append(f"{filename}|{text}|{json.dumps(codes)}\n")
            except Exception as exc:
                print(f"🦜 Lỗi encode file {filename}: {exc}")
                skipped_count += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines_to_write)

    print(f"🦜 Hoàn tất! Đã lưu file mã hóa tại: {output_path}")
    print(f"   - Tổng file xử lý thành công: {len(lines_to_write)}")
    print(f"   - Số file lỗi/bỏ qua: {skipped_count}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    encode_dataset(dataset_dir=os.path.join(project_root, "finetune", "dataset"))
