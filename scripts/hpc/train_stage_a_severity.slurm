#!/bin/bash -l
#
# Stage A retrain on NHR@FAU TinyGPU (rtx3080 partition, 1 GPU).
# Session 20, Round 3, Phase 4: same canonical hyperparameters as
# checkpoints/stage_a_combo (train_stage_a_combo.slurm) -- retrained ONLY on
# the severity-balanced rebuild of data/processed/stage_a (low/medium/high
# severity now present and balanced per class). Single-variable comparison,
# same precedent as every prior rebuild-then-retrain this session. Writes to
# a NEW checkpoint dir so stage_a_combo stays available for comparison.
#
#SBATCH --job-name=stage_a_severity
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=6:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate stage_a

python3 scripts/train_stage_a.py \
    --data data/processed/stage_a \
    --weights weights/yolo11m.pt \
    --epochs 20 \
    --batch-size 16 \
    --lr 1e-3 \
    --img-size 640 \
    --device cuda \
    --out checkpoints/stage_a_severity
