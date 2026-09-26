#!/bin/bash -l
#
# Stage B faint-distortion variant (Session 22): identical to
# train_stage_b_gtfix.slurm except --low-severity-weight 3.0 -- training
# images containing a low-severity distortion are sampled 3x as often
# (labels unchanged), to test whether it improves detection of faint
# distortions. Override the partition at submit time, e.g. `sbatch -p v100 ...`.
#
#SBATCH --job-name=stage_b_faint
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
    --low-severity-weight 3.0 \
    --device cuda \
    --out checkpoints/stage_b_faint
