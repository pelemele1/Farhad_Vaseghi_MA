#!/bin/bash -l
#
# Rebuild the Stage B dataset with balanced low/medium/high severity levels
# per class (Session 20, Round 3, supervisor request), carrying forward the
# recalibrated dirt/water/scratch tile-coverage thresholds from the
# previous round. CPU-only, no GPU needed, but TinyGPU requires
# --gres=gpu:1 on every job regardless.
#
#SBATCH --job-name=build_b_severity
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=3:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate stage_a

python3 scripts/build_stage_b_dataset.py \
    --source data/raw/mio_tcd/images \
    --out data/processed/stage_b \
    --variants 14 --include-combos --include-severity \
    --dirt-threshold 0.20 --water-threshold 0.25 --scratch-threshold 0.015
