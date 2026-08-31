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
# --loss focal, epochs 20->40 (Session 17): the first real run (job 1799124,
# default --loss bce) scored well on dirt/water (AP ~0.86) but badly on
# scratch (AP 0.309, precision 0.063) -- pos_weight~95 for scratch's ~1%
# tile-positive rate pushed the head to over-predict "scratch" almost
# everywhere rather than localize it. Switched to focal loss (Lin et al.
# 2017), architecture.md's own listed alternative for class imbalance, and
# doubled epochs since train/val loss hadn't plateaued at 20. See
# docs/development_log.md Sessions 16-17 for the full comparison.
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
    --epochs 40 \
    --batch-size 16 \
    --lr 1e-3 \
    --img-size 512 \
    --loss focal \
    --device cuda \
    --out checkpoints/stage_b
