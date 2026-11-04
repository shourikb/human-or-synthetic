#!/bin/bash

set -euo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: bash tuning/scripts/sweep_dpo.sh <model> <dataset> <train_size> <results_csv> [extra tuning.run_dpo flags]" >&2
    exit 1
fi

LEARNING_RATES=(2e-4 1e-4 5e-5)
BATCH_SIZES=(1 2)
GRAD_ACCUM=(4 8)

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=$1
DATASET=$2
TRAIN_SIZE=$3
RESULTS_CSV=$4

run_point() {
    local lr=$1 bs=$2 ga=$3
    echo "=== lr=$lr per_device_batch_size=$bs grad_accum=$ga (effective batch size $((bs * ga))) ==="
    python -m tuning.run_dpo \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --train_size "$TRAIN_SIZE" \
        --learning_rate "$lr" \
        --per_device_train_batch_size "$bs" \
        --gradient_accumulation_steps "$ga" \
        --results_csv "$RESULTS_CSV" \
        "${@:4}"
}

POINTS=()
for lr in "${LEARNING_RATES[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        for ga in "${GRAD_ACCUM[@]}"; do
            POINTS+=("$lr $bs $ga")
        done
    done
done

if [ -n "${TASK_ID:-}" ]; then
    run_point ${POINTS[$TASK_ID]} "${@:5}"
else
    for point in "${POINTS[@]}"; do
        run_point $point "${@:5}"
    done
fi
