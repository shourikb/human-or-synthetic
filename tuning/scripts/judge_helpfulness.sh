#!/bin/bash
#SBATCH --job-name=judgingresponses
#SBATCH --output=judgingresponses_%A.out
#SBATCH --error=judgingresponses_%A.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --nodes=1 --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --exclude="clippy,xaea-12,sonny"

export PYTHONUNBUFFERED=TRUE
export CUDA_VISIBLE_DEVICES=0

source ~/.bashrc
conda activate winrate
cd /coc/pskynet6/sbanerjee332/dpo-training

echo "Node:          $SLURMD_NODENAME"

MODEL="seed-55-h-555/responses_models555_qwen2.5-sft-a-400_pt-anthropic-307_54993.jsonl"

python judge_responses.py --judge_model openai/gpt-oss-20b \
    --trained_file ${MODEL} \
    --base_file datasets/anthropic_test.jsonl \
    --batch_size 64 \
    --dataset_type human \
    --output_file judge_results1.jsonl