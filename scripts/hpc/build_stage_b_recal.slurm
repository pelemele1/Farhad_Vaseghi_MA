#!/bin/bash -l
#
# Rebuild the Stage B dataset with recalibrated dirt/water tile thresholds
# (Session 20, Round 2, Phase 8) -- CPU-only, no GPU needed. Moved here from
# a local run after the water-droplet effect's skimage PiecewiseAffineTransform
# warp turned out to be far too slow on the local dev machine (~850/8000
# images in 40 minutes there).
#
#SBATCH --job-name=build_b_recal
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate stage_a

python3 scripts/build_stage_b_dataset.py \
    --source data/raw/mio_tcd/images \
    --out data/processed/stage_b_scratch15 \
    --variants 8 --include-combos \
    --dirt-threshold 0.20 --water-threshold 0.25 --scratch-threshold 0.015
