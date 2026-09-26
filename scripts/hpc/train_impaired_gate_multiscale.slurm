#!/bin/bash -l
#
# Impaired-gate v2 (Session 22): GAP over the stride-4/8/16 layers + P5
# (--arch multiscale), trained at 512x512 on the Stage B/C dataset's own
# images (same resolution the gate is applied at there; the v1 gate was
# trained at 640 on data/processed/stage_a). Same 14 kinds/split per source
# photo either way. Override the partition at submit time.
#
#SBATCH --job-name=gate_multiscale
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=6:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate stage_a

python3 -u scripts/train_impaired_gate.py \
    --data data/processed/stage_b \
    --weights weights/yolo11m.pt \
    --arch multiscale \
    --epochs 20 \
    --batch-size 16 \
    --lr 1e-3 \
    --img-size 512 \
    --num-workers 4 \
    --device cuda \
    --out checkpoints/impaired_gate_multiscale
