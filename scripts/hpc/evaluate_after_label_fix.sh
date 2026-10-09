#!/bin/bash
# Session 26: test-split evaluation of the label-fixed Stage B/C models with the label-fixed
# gate at its new recall-target threshold, clean-image false alarms, then all report figures.
#   GATE_T=<threshold> bash scripts/hpc/evaluate_after_label_fix.sh
set -e
: "${GATE_T:?set GATE_T to the gate threshold printed by gate_lf}"
G="--gate-checkpoint checkpoints/impaired_gate_multiscale_lf/impaired_gate_head.pt --gate-threshold $GATE_T"
python3 -u scripts/evaluate_stage_b.py --checkpoint checkpoints/stage_b_h32_lf/stage_b_head.pt \
    --data data/processed/stage_b --split test --tune-thresholds --by-severity $G --device cuda
python3 -u scripts/diagnose_clean_false_positives.py --checkpoint checkpoints/stage_b_h32_lf/stage_b_head.pt \
    --data data/processed/stage_b --split test --tune-thresholds $G --device cuda
python3 -u scripts/evaluate_stage_c.py --checkpoint checkpoints/stage_c_unet_t2_lf/stage_c_head.pt \
    --data data/processed/stage_b --split test --tune-thresholds --by-severity $G --device cuda
GATE_T=$GATE_T bash scripts/hpc/regenerate_report_figures.sh
