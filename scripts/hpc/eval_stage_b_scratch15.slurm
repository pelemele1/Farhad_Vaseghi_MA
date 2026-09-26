#!/bin/bash -l
#
# One-off evaluation of checkpoints/stage_b/stage_b_head.pt
# (Session 18, improvement #3 retrain) against its own held-out test split.
#
#SBATCH --job-name=eval_s15
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate stage_a

python3 scripts/evaluate_stage_b.py \
    --checkpoint checkpoints/stage_b/stage_b_head.pt \
    --data data/processed/stage_b \
    --split test \
    --device cuda
