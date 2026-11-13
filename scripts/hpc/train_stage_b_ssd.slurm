#!/bin/bash -l
#
# Stage B training on NHR@FAU TinyGPU (rtx3080 partition, 1 GPU).
# Session 19+, supervisor item 2: localized (per-tile) sum of squared
# differences loss (LocalizedSSDLoss, src/models/losses.py), compared
# against the current focal alpha=0.75, gamma=2.0 winner
# (checkpoints/stage_b) on the SAME dataset
# (data/processed/stage_b) for a clean single-variable
# comparison. No imbalance-correction hyperparameters -- plain SSD as
# literally requested.
#
#SBATCH --job-name=stage_b_ssd
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=6:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate stage_a

python3 scripts/train_stage_b.py \
    --data data/processed/stage_b \
    --weights weights/yolo11m.pt \
    --epochs 40 \
    --batch-size 16 \
    --lr 1e-3 \
    --img-size 512 \
    --loss ssd \
    --device cuda \
    --out checkpoints/stage_b_ssd
