#!/bin/bash

set -euo pipefail

if [ $# -lt 6 ]; then
    echo "Usage: bash tuning/scripts/judge.sh <task> <judge> <responses_file> <test_file> <output_file> <raw_output_file> [extra flags]" >&2
    exit 1
fi

TASK=$1
JUDGE=$2

case "$TASK" in
    helpfulness|summarization) ;;
    *) echo "Unknown task '$TASK' (expected helpfulness or summarization)" >&2; exit 1 ;;
esac
case "$JUDGE" in
    gptoss) MODULE="tuning.evaluation.judge_${TASK}" ;;
    qwen)   MODULE="tuning.evaluation.judge_${TASK}_qwen" ;;
    *) echo "Unknown judge '$JUDGE' (expected gptoss or qwen)" >&2; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$5")" "$(dirname "$6")"

python -m "$MODULE" \
    --trained_file "$3" \
    --base_file "$4" \
    --output_file "$5" \
    --raw_output_file "$6" \
    --batch_size 64 \
    "${@:7}"
