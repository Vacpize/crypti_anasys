#!/usr/bin/env bash
# Runs a python script from the repo inside the crypti-train container on DGX Spark.
# Usage (from the repo root): bash experiments/spark_run.sh ft_train.py --model Qwen/Qwen3.5-4B-Base ...
set -euo pipefail
cd "$(dirname "$0")/.."
exec docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -v "$HOME/.cache/huggingface:/hf" -e HF_HOME=/hf \
  -v "$PWD:/work" -w /work --network host \
  crypti-train python3 "$@"
