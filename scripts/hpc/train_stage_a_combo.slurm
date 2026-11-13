#!/bin/bash -l
#
# Stage A training on NHR@FAU TinyGPU (rtx3080 partition, 1 GPU).
# Session 19+, supervisor item 5: retrain on the rebuilt Stage A dataset
# (data/processed/stage_a), which now includes multi-distortion (combo)
# variants -- 8000 images total (8 kinds/source image: clean, 3 single
# effects, 3 pairs, the full triple), vs. the original 4000-image
# single-distortion-only dataset. Same config as the original Session 10
# run (train_stage_a.slurm) for a clean before/after comparison -- only the
# dataset changes. Writes to a NEW checkpoint dir (not overwriting
# checkpoints/stage_a in place) so the pre-combo result stays available for
# comparison in the final report.
#
#SBATCH --job-name=stage_a_combo
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
    --out checkpoints/stage_a_combo
