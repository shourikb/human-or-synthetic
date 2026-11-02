#!/bin/bash
#SBATCH --job-name=generatingresponses
#SBATCH --output=generatingresponses_%A.out
#SBATCH --error=generatingresponses_%A.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --nodes=1 --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --exclude="clippy,xaea-12,sonny,ig-88,kitt"

export PYTHONUNBUFFERED=TRUE

source ~/.bashrc
conda activate pydpo
cd /coc/pskynet6/sbanerjee332/dpo-training

echo "Node:          $SLURMD_NODENAME"

MODEL="models555/qwen2.5-sft-a-400_pt-anthropic-307_54993" # "unsloth/Meta-Llama-3.1-8B" 

# Current seed is 55

python generate_responses2.py \
    --model "$MODEL" \
    --model_name qwen2.5-7B \
    --input_file datasets/anthropic_test.jsonl \
    --num_samples 2048 \
    --batch_size 64