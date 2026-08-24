#!/bin/bash -l
#
# Stage A training on NHR@FAU TinyGPU (a100 partition, 1 GPU).
# Submit from the repo root with: sbatch.tinygpu scripts/hpc/train_stage_a.slurm
#
#SBATCH --job-name=stage_a
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
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
    --batch-size 32 \
    --lr 1e-3 \
    --img-size 640 \
    --device cuda \
    --out checkpoints/stage_a
