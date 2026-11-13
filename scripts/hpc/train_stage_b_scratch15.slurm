#!/bin/bash -l
#
# Stage B training on NHR@FAU TinyGPU (rtx3080 partition, 1 GPU).
# Session 18, improvement #3: rebuilt Stage B dataset with the scratch tile
# threshold lowered 0.03 -> 0.015 (see scripts/diagnose_scratch_threshold.py:
# median coverage of tiles the scratch mask actually touches is 0.0172, so
# the old 0.03 cutoff discarded over half of real scratch tiles as
# negatives). Same winning loss config as checkpoints/stage_b_alpha75
# (focal, alpha=0.75, gamma=2.0) for a clean single-variable comparison --
# only the dataset changes. Original dataset (data/processed/stage_b) is
# kept untouched; this trains against the new copy
# (data/processed/stage_b) so both remain comparable/reproducible.
#
#SBATCH --job-name=stage_b_s15
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
    --loss focal \
    --focal-alpha 0.75 \
    --focal-gamma 2.0 \
    --device cuda \
    --out checkpoints/stage_b
