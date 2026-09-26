#!/bin/bash -l
#
# Impaired-gate head retrain on NHR@FAU TinyGPU (rtx3080 partition, 1 GPU).
# Session 20, Round 3, Phase 4: same setup as checkpoints/impaired_gate
# (train_impaired_gate.slurm) -- image-level binary "is this image impaired
# at all" classifier, trained with CrossEntropyLoss -- retrained ONLY on the
# severity-balanced rebuild of data/processed/stage_a. Worth a dedicated
# look at whether low-severity images are harder for the gate to call
# "impaired" (use --by-severity in scripts/evaluate_stage_a.py at eval
# time). Writes to a NEW checkpoint dir so impaired_gate stays available for
# comparison.
#
#SBATCH --job-name=impaired_gate_severity
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=6:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate stage_a

python3 scripts/train_impaired_gate.py \
    --data data/processed/stage_a \
    --weights weights/yolo11m.pt \
    --epochs 20 \
    --batch-size 16 \
    --lr 1e-3 \
    --img-size 640 \
    --device cuda \
    --out checkpoints/impaired_gate_severity
