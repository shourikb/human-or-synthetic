#!/bin/bash
#SBATCH --job-name=verifytemplates
#SBATCH --output=verifytemplates_%A.out
#SBATCH --error=verifytemplates_%A.err
#SBATCH --partition=nlprx-lab
#SBATCH --nodes=1 --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --exclude="clippy,xaea-12,sonny"

export PYTHONUNBUFFERED=TRUE
export CUDA_VISIBLE_DEVICES=0

source ~/.bashrc
conda activate pydpo
cd /coc/pskynet6/sbanerjee332/dpo-training

python verify_templates.py