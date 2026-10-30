# syntax=docker/dockerfile:1
# Submission image: PhoWhisper-large fine-tune (ASR) + Qwen3-4B-Instruct-2507 with the LoRA merged in (vLLM).
# Build with docker/build.sh; weights come from a named build context laid out like Drive MyDrive/zalo_lifemate:
#   docker build --build-context weights=<dir> --build-arg ADAPTER=lora_para1 -t lifemate .
# v0.32.0 is the vLLM that colab_llm.ipynb installed for the scored runs; CUDA 13 needs driver >= 580,
# use vllm/vllm-openai:v0.32.0-x86_64-cu129 (driver >= 575) if the organisers' driver is older.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.32.0-x86_64

# ---- merge the LoRA adapter into the base model (CPU, bf16); peft and the base weights stay in this stage ----
FROM ${VLLM_IMAGE} AS merge
ARG ADAPTER
RUN test -n "${ADAPTER}" || { echo "pass --build-arg ADAPTER=<adapter dir in the weights context>"; exit 1; }
RUN uv pip install --system peft==0.21.2
COPY scripts/merge_lora.py /merge_lora.py
COPY --from=weights ${ADAPTER} /adapter
RUN python3 /merge_lora.py --adapter /adapter --out /merged

# ---- inference image ----
FROM ${VLLM_IMAGE}
WORKDIR /app
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt
# paths match configs/baseline.json (relative to /app)
COPY --from=weights models/phowhisper_large_ft_fp16 models/phowhisper_large_ft
COPY --from=merge /merged models/qwen3-4b-sft-merged
COPY . .
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
ENTRYPOINT ["bash", "/app/docker/run.sh"]
