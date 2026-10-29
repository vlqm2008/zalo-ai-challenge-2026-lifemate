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
