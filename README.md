# LifeMate baseline (Zalo AI Challenge 2026)

Pipeline: giọng nói tiếng Việt → ASR (PhoWhisper-large fine-tune) → agent Qwen3-4B-Instruct-2507 (LoRA SFT) gọi tool
từng bước trên `zaic_tools.ToolEnv` → `submission.jsonl`.

Gói này chỉ có mã nguồn, config và hướng dẫn. Không kèm dữ liệu cuộc thi, trọng số model hay dữ liệu tự sinh.

## Cài đặt

```bash
pip install -r requirements.txt        # vllm, transformers, peft, trl, soundfile, librosa ...
export PYTHONPATH=$PWD/src
```

Đặt dữ liệu của ban tổ chức ở `data/train/` và `data/public_test/` (giữ nguyên cấu trúc gốc, kể cả `zaic_tools`).
Không đọc trực tiếp thư mục `*_environment/`; mọi dữ liệu app chỉ lấy qua `zaic_tools`.

## 1. ASR: PhoWhisper-large fine-tune trên clip train

`splits/train_ids.txt` / `splits/dev_ids.txt` tách clip train để chấm ASR trên phần chưa train.

```bash
# vòng 1: toàn bộ model, 3 epoch
python scripts/finetune_asr.py --data ./data --split train --model vinai/PhoWhisper-large \
    --out models/phowhisper_large_ft --epochs 3 --batch 4 --lr 1e-5 --bf16 --eval-every 800 --eval-n 300
# vòng 2 (tuỳ chọn): đóng băng encoder trừ 16 lớp cuối, lr 5e-6, 2 epoch
python scripts/finetune_asr.py --data ./data --split train --model models/phowhisper_large_ft \
    --out models/phowhisper_large_ft2 --freeze-encoder --unfreeze-last-k 16 --bf16 --batch 2 --lr 5e-6 --epochs 2
```

Kết quả tham khảo trên 959 clip dev chưa train: WER ≈ 0.097, số tiền đúng ≈ 86 %.

Có thể nhận dạng trước một lần cho cả split rồi dùng cache (nhanh hơn khi thử nhiều LLM). Chạy mỗi GPU một tiến trình
riêng (`--gpus 1`, `--ids-file` cho từng phần), không nạp nhiều model trong cùng một tiến trình:

```bash
python scripts/precompute_asr.py --data ./data --split public_test --backend hf --model models/phowhisper_large_ft \
    --prompt domain --gpus 1 --no-metrics --out out/asr_cache_public_test.json
```

## 2. Dữ liệu SFT

```bash
python scripts/build_sft.py --data ./data --split train --ids-file splits/train2_ids.txt --out out/sft_train.jsonl
```

Mỗi dòng là một chuỗi hội thoại đúng định dạng lúc chạy: system prompt, user message (CONTEXT, HISTORY, yêu cầu, HINTS
tính bằng code, TOOLS), rồi từng lượt assistant là đúng một lệnh gọi JSON `{"name", "arguments"}`, kết quả tool, …,
`finish`. `splits/dev2_ids.txt` (300 mẫu, tách theo khuôn câu) là tập đánh giá; không train trên nó.

## 3. LoRA SFT cho Qwen3-4B

```bash
accelerate launch --num_processes 4 --mixed_precision bf16 scripts/train_lora.py \
    --train out/sft_train.jsonl --out out/lora --epochs 2 --lr 1e-4 --r 16 --max-len 10240 \
    --dtype bfloat16 --grad-accum 4 --assistant-only
python scripts/merge_lora.py --adapter out/lora --out models/qwen3-4b-sft-merged
```

`--assistant-only` chỉ tính loss trên các lượt assistant (quan trọng: train cả chuỗi kém hơn rõ).

## 4. Chạy và chấm

```bash
# chấm trên dev2 (transcript chuẩn hoặc cache ASR)
python scripts/run_dev.py --data ./data --split train --ids-file splits/dev2_ids.txt --oracle \
    --config configs/baseline.json --out-dir out/dev2_oracle
python scripts/error_buckets.py --run out/dev2_oracle        # phân loại lỗi

# tạo file nộp
python scripts/infer.py --data ./data --split public_test --config configs/baseline.json \
    --asr-cache out/asr_cache_public_test.json --out out/public/submission.jsonl
python scripts/check_submission.py --data ./data --split public_test --sub out/public/submission.jsonl
```

Mỗi mẫu dùng đúng một `ToolEnv`; decode tất định (temperature 0, seed 0, batch 1); có guided JSON trong vLLM.

## 5. Docker nộp bài

Ảnh chứa PhoWhisper fine-tune (fp16) và Qwen3-4B đã gộp LoRA (bf16, gộp lúc build từ `--build-arg ADAPTER`), chạy
offline. Khi chạy, `docker/run.sh` nhận dạng toàn bộ audio của split (`precompute_asr.py`, không prompt) rồi chạy
`infer.py` với `configs/baseline.json`.

Giả định tạm, sửa khi BTC ra hướng dẫn Docker: split mount ở `/data` (chính thư mục split hoặc thư mục chứa nó), file
nộp ghi ra `/result/submission.jsonl` (đổi bằng `-e DATA=... -e OUT=...`), log từng mẫu ở `/result/logs/`; base
`vllm/vllm-openai:v0.32.0-x86_64` (đúng bản vLLM Colab đã dùng; CUDA 13 cần driver ≥ 580, nếu driver BTC cũ hơn thì
thêm `--build-arg VLLM_IMAGE=vllm/vllm-openai:v0.32.0-x86_64-cu129`, driver ≥ 575); file nén gzip, in cả sha256 và md5.

Ước tính: ảnh ~31 GB (base ~20 GB + trọng số 11 GB), file `.tar.gz` ~20 GB; `build.sh` in con số thật. Drive miễn phí
chỉ có 15 GB, nên tài khoản upload cần gói dung lượng lớn hơn.

**Máy build và test.** Mac không chạy CUDA, Colab không chạy Docker: thuê một máy ảo GPU (VM thật, có Docker và
NVIDIA Container Toolkit), ví dụ Lambda Cloud 1×A100 40 GB. A100 cùng kiến trúc sm_80 với A30, còn vLLM đã giới hạn
14 GB nên bộ nhớ dùng như trên A30. Pod RunPod hay container Vast.ai thường không chạy được Docker bên trong. Cần ≥ 150 GB
đĩa trống, tổng khoảng 3 giờ máy.

```bash
# 1. chuẩn bị (rclone: tạo remote "gdrive"; máy không có trình duyệt thì chạy `rclone authorize "drive"` trên Mac rồi dán token)
nvidia-smi && docker buildx version
sudo apt-get install -y rclone pigz && rclone config
git clone https://$GH_TOKEN@github.com/FirstKhoi/zalo_lifemate.git && cd zalo_lifemate   # nhánh có Dockerfile

# 2. trọng số và dữ liệu từ Drive MyDrive/zalo_lifemate
rclone copy gdrive:zalo_lifemate/models/phowhisper_large_ft_fp16 ~/zl_weights/models/phowhisper_large_ft_fp16 -P
rclone copy gdrive:zalo_lifemate/lora_para1 ~/zl_weights/lora_para1 -P
rclone copy gdrive:zalo_lifemate/train+public_test.zip ~ -P && unzip -q ~/train+public_test.zip -d ~

# 3. build, lưu out/lifemate_lora_para1.tar.gz, in checksum
docker/build.sh ~/zl_weights lora_para1

# 4. chạy offline trên public_test, như BTC chạy
mkdir -p ~/result
time docker run --rm --gpus '"device=0"' --network none \
    -v ~/train+public_test/public_test:/data:ro -v ~/result:/result lifemate:lora_para1

# 5. kiểm tra file nộp (cần "OK: submission passes all checks"; p95 × 2 cho A30 phải < 30 s)
docker run --rm --entrypoint python3 -v ~/train+public_test/public_test:/data:ro -v ~/result:/result \
    lifemate:lora_para1 scripts/check_submission.py --data /data --split public_test --sub /result/submission.jsonl

# 6. so với submission.jsonl Colab của cùng adapter (chép từ Drive runs/<run>/): gần như mọi state_hash phải trùng
python3 - ~/result/submission.jsonl ~/colab_submission.jsonl <<'EOF'
import json, sys
load = lambda p: {r["id"]: r["state_hash"] for r in map(json.loads, open(p))}
a, b = load(sys.argv[1]), load(sys.argv[2])
print(sum(a[k] == b.get(k) for k in a), "/", len(a), "same state_hash")
EOF

# 7. file nén nạp lại được, rồi upload và lấy link công khai
docker image rm lifemate:lora_para1 && docker load -i out/lifemate_lora_para1.tar.gz
rclone copy out/lifemate_lora_para1.tar.gz gdrive:zalo_lifemate_submit -P
rclone link gdrive:zalo_lifemate_submit/lifemate_lora_para1.tar.gz
```

Điền link và checksum (`out/*.sha256` hoặc `.md5`, tuỳ portal) lên portal. Có thể nộp `~/result/submission.jsonl` lên
leaderboard public để chắc điểm khớp bản Colab. Xong thì xoá (terminate) máy ảo, vì trên đó có bộ dữ liệu.

## Những điểm đã đo được

| Thay đổi | Ảnh hưởng |
|---|---|
| Base Qwen3-4B, không fine-tune | public ≈ 34 |
| LoRA SFT trên gold train, loss chỉ trên assistant | public ≈ 43 |
| Không ép trường "thought" trong JSON (SFT không có thought) | dev2 +4.9 |
| Thêm dữ liệu tổng hợp đã chạy thật trên môi trường | public ≈ 56 |
| Sửa khung agent (chốt chặn số tiền, trùng tên, gợi ý giờ tiếng Việt) | +0.7 public |
| Masking tên hàm kiểu Hammer, DPO từng bước, ASR thêm ViMD | không giúp hoặc giảm điểm |

Tỉ lệ điểm dev2 (ASR) giữa hai bản dự đoán khá sát chênh lệch điểm public.
