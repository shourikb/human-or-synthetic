#!/bin/bash
#SBATCH --job-name=dpo_train
#SBATCH --output=dpo_train_%A.out
#SBATCH --error=dpo_train_%A.err
#SBATCH --partition="nlprx-lab"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="long"
#SBATCH --exclude="clippy,xaea-12,sonny,cyborg,ig-88"

export PYTHONUNBUFFERED=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/.bashrc
conda activate pydpo
cd /coc/pskynet6/sbanerjee332/dpo-training

# ----------------------------
# Hyperparameter grids
# ----------------------------
LEARNING_RATES=(2e-4) # (2e-4 1e-4 5e-5)
BATCH_SIZES=(1) # (1 2)
GRAD_ACCUM=(4)  # (4 8)

NUM_LR=${#LEARNING_RATES[@]}
NUM_BS=${#BATCH_SIZES[@]}
NUM_GA=${#GRAD_ACCUM[@]}

# ----------------------------
# Map SLURM array index -> params
# ----------------------------
IDX=$SLURM_ARRAY_TASK_ID

LR_IDX=$(( IDX / (NUM_BS * NUM_GA) ))
BS_IDX=$(( (IDX / NUM_GA) % NUM_BS ))
GA_IDX=$(( IDX % NUM_GA ))

LR=${LEARNING_RATES[$LR_IDX]}
BS=${BATCH_SIZES[$BS_IDX]}
GA=${GRAD_ACCUM[$GA_IDX]}

EFFECTIVE_BS=$(( BS * GA ))

echo "----------------------------------------"
echo "SLURM TASK ID: $IDX"
echo "Node:          $SLURMD_NODENAME"
echo "Learning rate: $LR"
echo "Batch size:    $BS"
echo "Grad accum:    $GA"
echo "Effective BS:  $EFFECTIVE_BS"
echo "----------------------------------------"

# ----------------------------
# Run training
# ----------------------------
python run_dpo.py \
  --model llama3-sft-a-1000 \
  --dataset anthropic \
  --train_size 769 \
  --learning_rate $LR \
  --per_device_train_batch_size $BS \
  --gradient_accumulation_steps $GA