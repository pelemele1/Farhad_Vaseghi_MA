#!/bin/bash -l
#
# Impaired-gate head training on NHR@FAU TinyGPU (rtx3080 partition, 1 GPU).
# Session 20, supervisor item 3a: image-level binary "is this image impaired
# at all" classifier, trained with CrossEntropyLoss, distinct from Stage A's
# 3-class per-defect head. Trains on the existing data/processed/stage_a
# (combo) dataset -- no dataset rebuild, the "impaired" label is derived on
# the fly (see src/data/stage_a_dataset.py::ImpairedGateDataset). Writes to
# checkpoints/impaired_gate, a brand-new checkpoint dir.
#
#SBATCH --job-name=impaired_gate
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
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
    --out checkpoints/impaired_gate
