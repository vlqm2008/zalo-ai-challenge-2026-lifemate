#!/usr/bin/env bash
# Container entrypoint: ASR for the whole mounted split, then the agent writes the submission.
# Assumed I/O until the organisers' Docker guide is out: split mounted at /data, output /result/submission.jsonl.
set -euo pipefail
cd /app
DATA=${DATA:-/data}
OUT=${OUT:-/result/submission.jsonl}
CACHE=/tmp/lifemate/asr_cache.json

# ASR runs in its own process so its GPU memory is free before vLLM starts
python3 scripts/precompute_asr.py --data "$DATA" --backend hf --model models/phowhisper_large_ft \
    --gpus 1 --no-metrics --out "$CACHE"
python3 scripts/infer.py --data "$DATA" --out "$OUT" --config configs/baseline.json --asr-cache "$CACHE"
