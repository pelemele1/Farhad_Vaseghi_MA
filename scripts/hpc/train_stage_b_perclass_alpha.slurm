#!/bin/bash -l
#
# Stage B training on NHR@FAU TinyGPU (rtx3080 partition, 1 GPU).
# Session 18, improvement #2: per-class focal-loss alpha instead of one
# shared value. alpha=0.75 (Session 18) already won on all three classes
# at the default threshold vs. bce and alpha=0.25 (see
# docs/stage_b_final_report.md) -- dirt/water are working well there, but
# scratch (the rarest, thinnest-signal class) is still the weak point
# (F1 ~0.42). FocalLossWithLogits already supports a per-class alpha
# tensor (src/models/losses.py); scripts/train_stage_b.py's --focal-alpha
# now accepts 3 comma-separated values (dirt,water,scratch) to use it.
# Keeps dirt/water at the winning 0.75 and pushes scratch specifically to
# 0.9, to test whether that recovers more of scratch's recall without
# disturbing dirt/water's already-good numbers.
#
#SBATCH --job-name=stage_b_pca
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
    --focal-alpha 0.75,0.75,0.9 \
    --focal-gamma 2.0 \
    --device cuda \
    --out checkpoints/stage_b_perclass_alpha
