#!/bin/bash
#SBATCH --job-name=prefsumgenerate
#SBATCH --output=prefjudgementssum_%a.out
#SBATCH --error=prefjudgementssum_%a.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="a40:4"
#SBATCH --qos=short
#SBATCH --account=overcap
#SBATCH --exclude=clippy,xaea-12,sonny,cyborg,ig-88,megazord
#SBATCH --array=0-11         # 12 tasks, one per run_id

export PYTHONUNBUFFERED=TRUE
export VLLM_SKIP_P2P_CHECK=1
source ~/.bashrc
conda activate pydpo
# cd /srv/nlprx-lab/share5/sbanerjee332
cd /coc/pskynet6/sbanerjee332

# Each array task maps directly to a run_id (0–5)
RUN_ID=$SLURM_ARRAY_TASK_ID

# All 4 GPUs on this node, indexed 0–3 locally
GPU_IDS="0,1,2,3"

echo "================================================"
echo "SLURM Job ID:     $SLURM_JOB_ID"
echo "SLURM Array Task: $SLURM_ARRAY_TASK_ID"
echo "Run ID:           $RUN_ID"
echo "Node:             $SLURMD_NODENAME"
echo "GPUs:             $GPU_IDS"
echo "================================================"

python -u prefgenerator_summary.py \
    --run_id=$RUN_ID \
    --total_runs=12 \
    --gpu_id=$GPU_IDS \
    --num_samples=32 \
    --batch_size=64 \
    --output_dir=/coc/pskynet6/sbanerjee332 \
    --global_offset=0