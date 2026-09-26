#!/bin/bash -l
#
# Stage B retrain on NHR@FAU TinyGPU (rtx3080 partition, 1 GPU).
# Session 20, Round 3, Phase 4: same canonical hyperparameters as
# checkpoints/stage_b_recal (train_stage_b_recal.slurm) -- focal loss,
# alpha=0.75 uniform, gamma=2.0, 40 epochs -- retrained ONLY on the
# severity-balanced rebuild of data/processed/stage_b (same
# recalibrated dirt/water/scratch tile thresholds as Round 2, now with
# low/medium/high severity present and balanced per class). Single-variable
# comparison, same precedent as every prior rebuild-then-retrain this
# session. Writes to a NEW checkpoint dir so stage_b_recal stays available
# for before/after comparison.
#
#SBATCH --job-name=stage_b_severity
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
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
    --loss focal \
    --focal-alpha 0.75 \
    --focal-gamma 2.0 \
    --device cuda \
    --out checkpoints/stage_b_severity
