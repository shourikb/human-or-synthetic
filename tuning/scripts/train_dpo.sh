#!/bin/bash

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: bash tuning/scripts/train_dpo.sh <model> <dataset> <train_size> [extra tuning.run_dpo flags]" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m tuning.run_dpo \
    --model "$1" \
    --dataset "$2" \
    --train_size "$3" \
    "${@:4}"
