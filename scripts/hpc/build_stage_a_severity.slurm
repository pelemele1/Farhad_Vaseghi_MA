#!/bin/bash -l
#
# Rebuild the Stage A dataset with balanced low/medium/high severity levels
# per class (Session 20, Round 3, supervisor request) -- CPU-only, no GPU
# needed, but TinyGPU requires --gres=gpu:1 on every job regardless (same
# pattern as build_stage_b_recal.slurm).
#
#SBATCH --job-name=build_a_severity
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=3:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate stage_a

python3 scripts/build_stage_a_dataset.py \
    --source data/raw/mio_tcd/images \
    --out data/processed/stage_a \
    --variants 14 --include-combos --include-severity
