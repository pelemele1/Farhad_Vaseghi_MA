#!/bin/bash -l
#
# Rebuild data/processed/stage_b_scratch15 in place, adding full-resolution
# per-class pixel masks (Session 21, Stage C prep) alongside the existing
# tile-grid ground truth -- same source images, seed, variants, thresholds
# as the current on-disk (severity-balanced) dataset, so images/splits/
# severities come out identical; only the new masks/ folder is added. This
# is Stage C's (pixel-level segmentation) ground truth, computed for free
# in the same synthesis pass that already builds this Stage B dataset --
# see src/soiling/dataset_builder.py::build_stage_b_dataset's
# save_pixel_masks docstring. CPU-only, no GPU needed, but TinyGPU requires
# --gres=gpu:1 on every job regardless.
#
#SBATCH --job-name=build_b_pixel_masks
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
    --out data/processed/stage_b_scratch15 \
    --variants 14 --include-combos --include-severity \
    --dirt-threshold 0.20 --water-threshold 0.25 --scratch-threshold 0.015 \
    --save-pixel-masks
