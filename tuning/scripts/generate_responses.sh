#!/bin/bash

set -euo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: bash tuning/scripts/generate_responses.sh <model_path> <model_name> <test_file> <output_file> [extra flags]" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

python -m tuning.inference.generate_responses \
    --model "$1" \
    --model_name "$2" \
    --input_file "$3" \
    --output_file "$4" \
    --num_samples 2048 \
    --batch_size 64 \
    "${@:5}"
