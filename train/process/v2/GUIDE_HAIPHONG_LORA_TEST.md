# Guide: test LoRA VieNeu-TTS với dữ liệu Hải Phòng

Bài test này chỉ xác nhận pipeline chạy end-to-end, không tối ưu chất lượng accent.

## Môi trường

```powershell
conda activate tts_5
```

Package quan trọng:

```text
torch 2.8.0+cu128
neucodec 0.0.5
torchao 0.13.0
```

`torchao==0.13.0` tương thích với PyTorch 2.8 hiện tại.

## Dữ liệu

```text
D:\hustmedia\python\tts\vie_neu\vimd_hp_candidates\
```

CSV ưu tiên:

```text
metadata_preferred_6_15s.csv
```

Các cột đã xác nhận:

```text
downloaded_file | transcript | speakerID | duration_sec
accent_rating | noise_rating | style_rating | decision | notes
```

Trong ViMD, Hải Phòng có `province_name=HaiPhong` và `province_code=15`.

Nếu `decision` đã được chấm, script chỉ lấy `giữ`. Nếu còn trống, script dùng sample hợp lệ trong CSV preferred.

## Chuẩn bị dataset

```powershell
conda activate tts_5
cd D:\hustmedia\python\tts\vie_neu\train
python test_train_haiphong.py --prepare
```

Output:

```text
D:\hustmedia\python\tts\vie_neu\train\data\dataset_haiphong_test\
```

Các file chính:

```text
raw_audio\
metadata.csv
metadata_cleaned.csv
metadata_encoded.csv
```

Luồng xử lý:

```text
CSV
→ metadata.csv dạng file_name|text
→ filter_data.py chính thức
→ NeuCodec encode_data.py chính thức
→ phonemize_with_dict
→ thống kê sample/speaker/duration
```

`filter_data.py` sẽ loại text có số, acronym hoặc không kết thúc bằng dấu câu hợp lệ. Vì vậy số sample sau filter có thể nhỏ hơn đầu vào.

## Lỗi quyền truy cập NeuCodec

Nếu gặp `403 Forbidden`, `GatedRepoError` hoặc `Cannot access gated repo`:

1. Mở [neuphonic/neucodec](https://huggingface.co/neuphonic/neucodec).
2. Đăng nhập Hugging Face và chấp nhận điều kiện truy cập model.
3. Đăng nhập token trong đúng env:

```powershell
conda activate tts_5
hf auth login
hf auth whoami
```

Sau đó chạy lại:

```powershell
python test_train_haiphong.py --prepare
```

Không ghi token vào source code.

## Training sanity test

Chỉ chạy sau khi `metadata_encoded.csv` có sample:

```powershell
python test_train_haiphong.py --train
```

Cấu hình:

```text
Base model: pnnbao-ump/VieNeu-TTS-0.3B
LoRA: config chính thức
max_steps: 30
batch size: 1
gradient accumulation: 1
logging_steps: 1
save_steps: 30
```

Checkpoint:

```text
D:\hustmedia\python\tts\vie_neu\train\output\haiphong_test\
```

Không có CUDA thì script chỉ prepare dataset, không train CPU nặng.

## Chạy cả hai bước

```powershell
python test_train_haiphong.py --all
```

Nên chạy `--prepare` trước và kiểm tra số sample encode thành công; chưa chạy `--train` nếu chưa xác nhận dataset.
