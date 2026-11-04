#!/bin/bash

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: bash generation/scripts/generate_helpfulness.sh <personas_file> <output_file> [extra flags]" >&2
    exit 1
fi

NUM_GPUS=${NUM_GPUS:-4}
BATCHES_PER_GPU=${BATCHES_PER_GPU:-125}

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

pids=()
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    python -m generation.generate_helpfulness \
        --personas_file "$1" \
        --output_file "$2" \
        --gpu_id "$gpu" \
        --batch_start $((gpu * BATCHES_PER_GPU)) \
        --batch_count "$BATCHES_PER_GPU" \
        "${@:3}" &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
