VieNeu-TTS v2 — thử nghiệm Nghệ An: LoRA 30 mẫu + EOS

Thư mục này độc lập với test_1 và pipeline train production. Nó chỉ lưu PEFT
adapter, không hợp nhất adapter vào trọng số base.

1) Kích hoạt môi trường

source /root/miniconda3/etc/profile.d/conda.sh
conda activate tts_5
cd /root/media_tech_ai/vie_neu

2) Train đúng 30 mẫu an toàn

python train/process/v2/test_2/train_lora_30_eos.py \
  --run-name nghean_v2_lora30_eos \
  --epochs 80 \
  --learning-rate 5e-6 \
  --eos-loss-weight 5 \
  --fast-gpu \
  --overwrite

Script sẽ dừng rõ ràng nếu sau filter, NeuCodec encode và token audit không đủ
đúng 30 mẫu có SPEECH_GENERATION_END. Nó không tự train với 20 hoặc 22 mẫu.

3) Test base + adapter runtime

python train/process/v2/test_2/test_lora_model.py \
  --adapter \
  /root/media_tech_ai/vie_neu/train/output/nghean_v2_lora30_eos/adapter

Có thể truyền reference và câu riêng:

python train/process/v2/test_2/test_lora_model.py \
  --adapter /root/media_tech_ai/vie_neu/train/output/nghean_v2_lora30_eos/adapter \
  --reference /root/media_tech_ai/vie_neu/train/data/dataset_nghean/valid/spk_37_0210/37_0283.wav \
  --ref-text "Nội dung đúng nguyên văn của file reference." \
  --text "Một câu thử giọng Nghệ An."

4) Output

train/output/nghean_v2_lora30_eos/
  dataset/
  adapter/
  training_report.json
  dataset_token_audit.csv
  valid_eos_audit/

train/process/v2/test_2/outputs/
  lora_test_01.wav ...
  eos_inference_report.json

Inference luôn nạp base model, PEFT adapter và NeuCodec riêng ở runtime.
Fragment không có EOS, bị lặp token hoặc vượt duration hợp lý sẽ bị bỏ; không
decode thành WAV runaway. Pipeline thử lại ở nhiệt độ thấp, sau đó tách câu
đệ quy; fragment atomic cuối cùng vẫn lỗi thì bỏ riêng fragment đó.

Kiểm tra syntax/import:

python -m py_compile train/process/v2/test_2/train_lora_30_eos.py train/process/v2/test_2/test_lora_model.py
python train/process/v2/test_2/train_lora_30_eos.py --help
python train/process/v2/test_2/test_lora_model.py --help
