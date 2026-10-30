#!/usr/bin/env bash
# Build the submission image, save it as out/lifemate_<adapter>.tar.gz and print its checksums.
#   docker/build.sh <weights dir> <adapter subdir> [extra docker build args...]
#   docker/build.sh ~/zl_weights lora_para1
#   docker/build.sh ~/zl_weights lora_para1 --build-arg VLLM_IMAGE=vllm/vllm-openai:v0.32.0-x86_64-cu129
# <weights dir> is laid out like Drive MyDrive/zalo_lifemate: models/phowhisper_large_ft_fp16/ and <adapter subdir>/.
set -euo pipefail
WEIGHTS=$(cd "${1:?weights dir}" && pwd)
ADAPTER=${2:?adapter subdir inside the weights dir, e.g. lora_para1}
shift 2
cd "$(dirname "$0")/.."
test -f "$WEIGHTS/$ADAPTER/adapter_model.safetensors" || { echo "no adapter at $WEIGHTS/$ADAPTER"; exit 1; }
test -f "$WEIGHTS/models/phowhisper_large_ft_fp16/model.safetensors" || { echo "no ASR weights in $WEIGHTS/models/phowhisper_large_ft_fp16"; exit 1; }

TAG=lifemate:${ADAPTER}
OUT=out/lifemate_${ADAPTER}.tar.gz
docker build --build-context weights="$WEIGHTS" --build-arg ADAPTER="$ADAPTER" -t "$TAG" "$@" .
docker image ls "$TAG"
mkdir -p out
GZ=$(command -v pigz || echo gzip)
docker save "$TAG" | "$GZ" -1 > "$OUT"
ls -lh "$OUT"
(cd out && sha256sum "$(basename "$OUT")" | tee "$(basename "$OUT").sha256" && md5sum "$(basename "$OUT")" | tee "$(basename "$OUT").md5")
echo "load with: docker load -i $(basename "$OUT")   (image tag $TAG)"
