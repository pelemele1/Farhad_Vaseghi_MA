#!/bin/bash -l
#
# Stage B retrain on NHR@FAU TinyGPU (rtx3080 partition, 1 GPU).
# Session 20, Round 2, Phase 8: same canonical hyperparameters as
# checkpoints/stage_b_combo (train_stage_b_combo.slurm) -- focal loss,
# alpha=0.75 uniform, gamma=2.0, 40 epochs -- retrained ONLY on the
# recalibrated-threshold rebuild of data/processed/stage_b_scratch15
# (dirt 0.15->0.20, water 0.15->0.25, scratch unchanged at 0.015). Single-
# variable comparison against stage_b_combo, same precedent as the earlier
# scratch-threshold-only retrain. Writes to a NEW checkpoint dir so
# stage_b_combo stays available for before/after comparison.
#
#SBATCH --job-name=stage_b_recal
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=6:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate stage_a

python3 scripts/train_stage_b.py \
    --data data/processed/stage_b_scratch15 \
    --weights weights/yolo11m.pt \
    --epochs 40 \
    --batch-size 16 \
    --lr 1e-3 \
    --img-size 512 \
    --loss focal \
    --focal-alpha 0.75 \
    --focal-gamma 2.0 \
    --device cuda \
    --out checkpoints/stage_b_recal
