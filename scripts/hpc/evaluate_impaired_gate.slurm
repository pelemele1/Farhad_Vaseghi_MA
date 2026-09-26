#!/bin/bash -l
#
# Impaired-gate evaluation on a GPU, including the val-tuned gate threshold
# that keeps >= RECALL (default 0.95) of impaired images -- the value to pass
# as --gate-threshold to the Stage B/C evaluators.
#
#SBATCH --job-name=eval_gate
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=1:00:00
#SBATCH --output=%x_%j.out

unset SLURM_EXPORT_ENV

module load python
conda activate stage_a

python3 -u scripts/evaluate_impaired_gate.py \
    --checkpoint checkpoints/impaired_gate_severity/impaired_gate_head.pt \
    --data data/processed/stage_a --split test --device cuda \
    --tune-thresholds --recall-target "${RECALL:-0.95}"
