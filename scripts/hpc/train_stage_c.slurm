#!/bin/bash -l
#
# Stage C (pixel-level segmentation) training on NHR@FAU TinyGPU (rtx3080
# partition, 1 GPU). architecture.md §2's third distortion-head stage --
# frozen backbone, only the new StageCDistortionHead is optimized (same
# Option 1 strategy as every other stage). Requires
# data/processed/stage_b to have been rebuilt with
# --save-pixel-masks first (scripts/hpc/build_stage_b_pixel_masks.slurm).
#
#SBATCH --job-name=stage_c
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=6:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate stage_a

python3 scripts/train_stage_c.py \
    --data data/processed/stage_b \
    --weights weights/yolo11m.pt \
    --epochs 20 \
    --batch-size 16 \
    --lr 1e-3 \
    --img-size 512 \
    --device cuda \
    --out checkpoints/stage_c
