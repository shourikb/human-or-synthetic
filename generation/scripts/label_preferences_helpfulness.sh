#!/bin/bash

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: bash generation/scripts/label_preferences_helpfulness.sh <input_file> <output_dir> [extra flags]" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export VLLM_SKIP_P2P_CHECK=1

mkdir -p "$2"

python -m generation.label_preferences_helpfulness \
    --input_file "$1" \
    --output_dir "$2" \
    --run_id "${RUN_ID:-0}" \
    --total_runs "${TOTAL_RUNS:-1}" \
    --gpu_id "${GPU_IDS:-0,1,2,3}" \
    --num_samples 32 \
    --batch_size 64 \
    "${@:3}"
