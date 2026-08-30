#!/bin/bash -l
#
# Stage B training on NHR@FAU TinyGPU (rtx3080 partition, 1 GPU).
# Submit from the repo root with: sbatch.tinygpu scripts/hpc/train_stage_b.slurm
#
# Same cluster/partition choice as train_stage_a.slurm (see that file's
# comment: this account's association has no GPU quota on a100). Reuses the
# same "stage_a" conda env -- Stage B needs no new dependencies, so a
# separate env would just be extra setup for nothing; run
# scripts/hpc/setup_env.sh once if that env doesn't exist yet.
#
#SBATCH --job-name=stage_b
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
    --epochs 20 \
    --batch-size 16 \
    --lr 1e-3 \
    --img-size 512 \
    --device cuda \
    --out checkpoints/stage_b
