#!/bin/bash
#SBATCH --job-name=generatingsummary
#SBATCH --output=generatingsummary_%A.out
#SBATCH --error=generatingsummary_%A.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --nodes=1 --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="a40:4"
#SBATCH --qos="short"
#SBATCH --exclude="clippy,xaea-12,sonny"

export PYTHONUNBUFFERED=TRUE

source ~/.bashrc
conda activate pydpo
# cd /srv/nlprx-lab/share5/sbanerjee332
cd /coc/pskynet6/sbanerjee332

srun --multi-prog run-multi-sum.conf
# srun -u python -u run-model-batch.py