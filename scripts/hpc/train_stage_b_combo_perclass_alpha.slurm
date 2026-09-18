#!/bin/bash -l
#
# Stage B training on NHR@FAU TinyGPU (rtx3080 partition, 1 GPU).
# Session 19+, user follow-up: re-try per-class focal alpha now that the
# combo dataset has shifted the tile-level class balance a lot --
# dirt=26%, water=35% positive (train split), both far less rare than the
# 13%/18% they were pre-combo; scratch is 3% (was ~1.5%), still rare but
# less extreme too (pos_weight 32 vs ~95 before). The uniform alpha=0.75
# canonical config was tuned for the OLD, more extreme imbalance -- this
# tests whether lowering dirt/water's alpha to a neutral 0.5 (now that
# they're much better represented) recovers some of their combo-rebuild
# regression (checkpoints/stage_b_combo: dirt F1 0.803->0.755, water
# 0.826->0.815), while leaving scratch's alpha unchanged at 0.75 to isolate
# the variable to dirt/water specifically (unlike Session 18's
# stage_b_perclass_alpha attempt, which pushed scratch UP and made things
# worse on the old, pre-combo dataset).
#
#SBATCH --job-name=stage_b_combo_pca
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=6:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate stage_a

python3 scripts/train_stage_b.py \
    --data data/processed/stage_b_scratch15 \
    --weights weights/yolo11m.pt \
    --epochs 40 \
    --batch-size 16 \
    --lr 1e-3 \
    --img-size 512 \
    --loss focal \
    --focal-alpha 0.5,0.5,0.75 \
    --focal-gamma 2.0 \
    --device cuda \
    --out checkpoints/stage_b_combo_perclass_alpha
