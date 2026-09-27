#!/bin/bash
# Regenerates every figure used by the three stage reports from the canonical checkpoints
# (inference only, no training). Run on a GPU node from $WORK/Farhad_Vaseghi_MA, e.g.
#   sbatch -p a100 --gres=gpu:a100:1 scripts/hpc/run_shell.slurm "bash scripts/hpc/regenerate_report_figures.sh"
set -e
G="--gate-checkpoint checkpoints/impaired_gate_multiscale/impaired_gate_head.pt --gate-threshold 0.140"
python3 -u scripts/visualize_stage_a_results.py --checkpoint checkpoints/stage_a_multiscale/stage_a_head.pt \
    --data data/processed/stage_a --split test --tune-thresholds --log-file stage_a_multiscale_1823220.out \
    --tag _multiscale --out-dir docs/images --device cuda
python3 -u scripts/visualize_stage_a_class_examples.py --checkpoint checkpoints/stage_a_multiscale/stage_a_head.pt \
    --data data/processed/stage_a --split test --device cuda --out-dir docs/images
python3 -u scripts/visualize_stage_b_results.py --checkpoint checkpoints/stage_b_h32/stage_b_head.pt \
    --data data/processed/stage_b --split test --tune-thresholds $G \
    --log-file stage_b_h32_1823489.out --tag _h32 --out-dir docs/images --device cuda
python3 -u scripts/visualize_stage_c_results.py --checkpoint checkpoints/stage_c_unet_t2/stage_c_head.pt \
    --data data/processed/stage_b --split test --tune-thresholds $G \
    --log-file stage_c_unet_t2_1823237.out --tag _unet_t2 --out-dir docs/images --device cuda
