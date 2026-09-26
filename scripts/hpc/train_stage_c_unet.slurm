#!/bin/bash -l
#
# Stage C v2 (Session 22): U-Net-style decoder with skip connections from
# the frozen backbone's stride-4/8/16 layers + P5 (--arch unet), batch-level
# Dice + BCE, severity-independent pixel masks
# (scripts/make_gt_severity_independent.py). Keeps the best-val-loss epoch.
# Override the partition at submit time; EPOCHS / OUT env vars override the
# defaults, e.g. `sbatch --export=NONE,EPOCHS=50,OUT=checkpoints/stage_c_unet50 ...`.
#
#SBATCH --job-name=stage_c_unet
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=6:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate stage_a

python3 -u scripts/train_stage_c.py \
    --data data/processed/stage_b_scratch15 \
    --weights weights/yolo11m.pt \
    --arch unet \
    --epochs ${EPOCHS:-25} \
    --batch-size 16 \
    --lr 1e-3 \
    --img-size 512 \
    --num-workers 4 \
    --device cuda \
    --out ${OUT:-checkpoints/stage_c_unet}
