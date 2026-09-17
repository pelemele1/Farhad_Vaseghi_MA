#!/bin/bash -l
#
# Stage B training on NHR@FAU TinyGPU (rtx3080 partition, 1 GPU).
# EXPERIMENTAL variant of train_stage_b.slurm (Session 18) -- not committed
# as the project default yet. Tests whether raising focal loss's alpha
# (0.25 -> 0.75) recovers the recall that collapsed across all three
# classes in job 1799134 (Session 17). Rationale: alpha=0.25 is the focal
# loss paper's own best-found value for RetinaNet/COCO's ~1:1000
# foreground:background imbalance; this project's tile imbalance is far
# milder (13%/18%/1% positive rates per class), so the same alpha may be
# overcorrecting here. alpha=0.75 weights the positive class 3x more than
# alpha=0.25 does (alpha_t = alpha*target + (1-alpha)*(1-target)) --
# roughly the mirror-image setting. gamma and epochs left unchanged from
# job 1799134 for a clean single-variable comparison.
#
#SBATCH --job-name=stage_b_a75
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
    --out checkpoints/stage_b_alpha75
