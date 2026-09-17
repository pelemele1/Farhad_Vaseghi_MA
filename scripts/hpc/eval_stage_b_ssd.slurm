#!/bin/bash -l
#
# Evaluate checkpoints/stage_b_ssd/stage_b_head.pt (Session 19+ SSD-loss
# comparison run) against its own test split, both at the default 0.5
# threshold and with per-class tuned thresholds.
#
#SBATCH --job-name=eval_ssd
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate stage_a

echo "=== default threshold=0.5 ==="
python3 scripts/evaluate_stage_b.py \
    --checkpoint checkpoints/stage_b_ssd/stage_b_head.pt \
    --data data/processed/stage_b_scratch15 \
    --split test \
    --device cuda

echo ""
echo "=== tuned per-class thresholds ==="
python3 scripts/evaluate_stage_b.py \
    --checkpoint checkpoints/stage_b_ssd/stage_b_head.pt \
    --data data/processed/stage_b_scratch15 \
    --split test \
    --device cuda \
    --tune-thresholds
