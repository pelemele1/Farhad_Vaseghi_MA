#!/bin/bash -l
#
# Stage B retrain on severity-independent ground truth (Session 22): same
# canonical hyperparameters as train_stage_b_severity.slurm (focal loss,
# alpha=0.75, gamma=2.0, 40 epochs) -- the only change is the tile labels,
# converted in place by scripts/make_gt_severity_independent.py. The run
# keeps the best-val-loss epoch (scripts/train_stage_a.py::fit). Override
# the partition at submit time, e.g. `sbatch -p v100 ...`.
#
#SBATCH --job-name=stage_b_gtfix
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=6:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate stage_a

python3 -u scripts/train_stage_b.py \
    --data data/processed/stage_b_scratch15 \
    --weights weights/yolo11m.pt \
    --epochs 40 \
    --batch-size 16 \
    --lr 1e-3 \
    --img-size 512 \
    --loss focal \
    --focal-alpha 0.75 \
    --focal-gamma 2.0 \
    --num-workers 4 \
    --device cuda \
    --out checkpoints/stage_b_gtfix
