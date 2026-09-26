#!/bin/bash -l
#
# Stage B training on NHR@FAU TinyGPU (rtx3080 partition, 1 GPU).
# Session 19+, supervisor item 5: retrain on the rebuilt Stage B dataset
# (data/processed/stage_b), which now includes multi-distortion
# (combo) variants -- 8000 images, scratch threshold still 1.5% (carried
# forward from the prior session's adopted fix). Uses the winning loss
# config from Phase 3's comparison (focal alpha=0.75, gamma=2.0 -- SSD did
# not beat it, see docs/stage_b_final_report.md). Writes to a NEW checkpoint
# dir (checkpoints/stage_b_combo), not overwriting checkpoints/stage_b_
# scratch15 in place, so the pre-combo result stays available for
# comparison.
#
#SBATCH --job-name=stage_b_combo
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
    --focal-alpha 0.75 \
    --focal-gamma 2.0 \
    --device cuda \
    --out checkpoints/stage_b_combo
