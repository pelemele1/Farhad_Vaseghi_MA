#!/bin/bash -l
#
# Stage A training on NHR@FAU TinyGPU (rtx3080 partition, 1 GPU).
# Submit from the repo root with: sbatch.tinygpu scripts/hpc/train_stage_a.slurm
#
# Originally targeted a100, but this account's association has no GPU quota
# there (job sat PD forever, reason AssocGrpGRES) -- switched to rtx3080,
# confirmed available via an interactive salloc test. RTX3080 has 10GB VRAM
# vs the A100's 40GB, so batch-size is lowered from 32 to 16 as a
# precaution against an out-of-memory error; raise it back up if a run
# completes comfortably and you want more throughput.
#
#SBATCH --job-name=stage_a
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
    --out checkpoints/stage_a
