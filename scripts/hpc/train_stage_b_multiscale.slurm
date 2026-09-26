#!/bin/bash -l
#
# Stage B v2 (Session 22): multi-scale tile head (--arch multiscale,
# stride-4/8/16 + P5 features pooled to the 16x16 tile grid). Otherwise
# identical to train_stage_b_gtfix.slurm (focal alpha=0.75, gamma=2.0, 40
# epochs, severity-independent labels) -- single-variable comparison.
# Override the partition at submit time, e.g. `sbatch -p a100 --gres=gpu:a100:1 ...`.
#
#SBATCH --job-name=stage_b_multiscale
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=6:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate stage_a

python3 -u scripts/train_stage_b.py \
    --data data/processed/stage_b_scratch15 \
    --weights weights/yolo11m.pt \
    --arch multiscale \
    --epochs 40 \
    --batch-size 16 \
    --lr 1e-3 \
    --img-size 512 \
    --loss focal \
    --focal-alpha 0.75 \
    --focal-gamma 2.0 \
    --num-workers 4 \
    --device cuda \
    --out checkpoints/stage_b_multiscale
