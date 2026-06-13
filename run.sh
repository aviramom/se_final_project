#!/usr/bin/env bash
# Usage: ./run.sh [model]
#
# Available models:
#   mock         – Mock models only, runs locally (no cluster, no GPU)
#   chatts       – ChatTS-8B on Slurm GPU node
#   qwen         – Qwen3-VL-8B-Instruct on Slurm GPU node
#   all          – All models available (both GPU models + mocks) on Slurm GPU node
#
# Defaults (override by setting the env var before calling the script):
#   CHATTS_MODEL_PATH   /home/aviramom/models/chatts-8b
#   QWEN_VL_MODEL_PATH  /home/aviramom/models/qwen3-vl-8b
#   SLURM_GPU_TYPE      rtx_3090

set -euo pipefail

MODEL="${1:-}"

if [[ -z "$MODEL" ]]; then
  echo "Usage: ./run.sh [mock|chatts|qwen|all]"
  echo ""
  echo "  mock    – Mock models, local execution (no cluster needed)"
  echo "  chatts  – ChatTS-8B on Slurm GPU node"
  echo "  qwen    – Qwen3-VL-8B-Instruct on Slurm GPU node"
  echo "  all     – Both GPU models available in the same session"
  exit 1
fi

# ── Slurm GPU defaults (override by setting env vars before calling this script) ──
: "${SLURM_PARTITION:=main}"
: "${SLURM_GPUS:=1}"
: "${SLURM_GPU_TYPE:=rtx_3090}"
: "${SLURM_CPUS:=4}"
: "${SLURM_MEM_GB:=24}"
: "${SLURM_TIME_LIMIT:=02:00:00}"
: "${CHATTS_MODEL_PATH:=/home/aviramom/models/chatts-8b}"
: "${QWEN_VL_MODEL_PATH:=/home/aviramom/models/qwen3-vl-8b}"
# Home-staged UCR archive — mounted on all nodes (unlike /cs/azencot_fsas, which
# is only on CS-lab nodes). Needed for the icl_ucr_* benchmarks.
: "${UCR_DATA_PATH:=/home/aviramom/ucr_data/Univariate_arff}"

STREAMLIT=".venv/bin/streamlit"

case "$MODEL" in

  mock)
    echo "► Starting in local mock mode (no cluster)"
    FMEVAL_RUNNER=mock \
      "$STREAMLIT" run fmeval/app/main.py
    ;;

  chatts)
    echo "► Starting with ChatTS-8B on Slurm (partition=$SLURM_PARTITION, gpu=$SLURM_GPU_TYPE)"
    FMEVAL_RUNNER=slurm \
    SLURM_PARTITION="$SLURM_PARTITION" \
    SLURM_GPUS="$SLURM_GPUS" \
    SLURM_GPU_TYPE="$SLURM_GPU_TYPE" \
    SLURM_CPUS="$SLURM_CPUS" \
    SLURM_MEM_GB="$SLURM_MEM_GB" \
    SLURM_TIME_LIMIT="$SLURM_TIME_LIMIT" \
    UCR_DATA_PATH="$UCR_DATA_PATH" \
    CHATTS_MODEL_PATH="$CHATTS_MODEL_PATH" \
      "$STREAMLIT" run fmeval/app/main.py
    ;;

  qwen)
    echo "► Starting with Qwen3-VL-8B on Slurm (partition=$SLURM_PARTITION, gpu=$SLURM_GPU_TYPE)"
    FMEVAL_RUNNER=slurm \
    SLURM_PARTITION="$SLURM_PARTITION" \
    SLURM_GPUS="$SLURM_GPUS" \
    SLURM_GPU_TYPE="$SLURM_GPU_TYPE" \
    SLURM_CPUS="$SLURM_CPUS" \
    SLURM_MEM_GB="$SLURM_MEM_GB" \
    SLURM_TIME_LIMIT="$SLURM_TIME_LIMIT" \
    UCR_DATA_PATH="$UCR_DATA_PATH" \
    QWEN_VL_MODEL_PATH="$QWEN_VL_MODEL_PATH" \
      "$STREAMLIT" run fmeval/app/main.py
    ;;

  all)
    echo "► Starting with all models on Slurm (partition=$SLURM_PARTITION, gpu=$SLURM_GPU_TYPE)"
    FMEVAL_RUNNER=slurm \
    SLURM_PARTITION="$SLURM_PARTITION" \
    SLURM_GPUS="$SLURM_GPUS" \
    SLURM_GPU_TYPE="$SLURM_GPU_TYPE" \
    SLURM_CPUS="$SLURM_CPUS" \
    SLURM_MEM_GB="$SLURM_MEM_GB" \
    SLURM_TIME_LIMIT="$SLURM_TIME_LIMIT" \
    UCR_DATA_PATH="$UCR_DATA_PATH" \
    CHATTS_MODEL_PATH="$CHATTS_MODEL_PATH" \
    QWEN_VL_MODEL_PATH="$QWEN_VL_MODEL_PATH" \
      "$STREAMLIT" run fmeval/app/main.py
    ;;

  *)
    echo "Unknown model: '$MODEL'"
    echo "Available: mock | chatts | qwen | all"
    exit 1
    ;;

esac
