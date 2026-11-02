#!/bin/bash
#SBATCH --job-name=qwen_judgingresponses
#SBATCH --output=judgingresponses_qwen_%A.out
#SBATCH --error=judgingresponses_qwen_%A.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --nodes=1 --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="a40:2"
#SBATCH --qos="short"
#SBATCH --exclude="clippy,xaea-12,sonny"

export PYTHONUNBUFFERED=TRUE
export CUDA_VISIBLE_DEVICES=0,1

source ~/.bashrc
conda activate winrate
cd /coc/pskynet6/sbanerjee332/dpo-training

echo "Node:          $SLURMD_NODENAME"

MODEL="seed-42-h-123/responses_models_qwen2.5-sft-s-60_pt-synthetic-n-5849_41527.jsonl"

python judge_responses_qwen.py --judge_model Qwen/Qwen3.8-27B \
    --trained_file ${MODEL} \
    --base_file datasets/anthropic_test.jsonl \
    --batch_size 64 \
    --dataset_type comp \
    --output_file judge_results6.jsonl