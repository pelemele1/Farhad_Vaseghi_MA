#!/bin/bash
# Session 26: retrain every model on the label-fixed data (scripts/fix_dataset_labels.py),
# with exactly the canonical settings of the reports, into *_lf checkpoint dirs (the old
# models stay for comparison). Each job trains, then evaluates on the test split; Stage B/C
# test runs with the new gate come afterwards, once its recall-target threshold is known.
#   bash scripts/hpc/retrain_after_label_fix.sh [dependency job id]
#   PART=v100 bash scripts/hpc/retrain_after_label_fix.sh     (uses the stage_a_v100 env)
set -e
DEP=${1:+--dependency=afterok:$1}
PART=${PART:-a100}
ENV=stage_a
[ "$PART" = v100 ] && ENV=stage_a_v100
mkdir -p study_logs
submit() {  # name, time, command
    sbatch.tinygpu -p "$PART" --gres=gpu:$PART:1 --time="$2" $DEP --export=NONE,CONDA_ENV=$ENV         --job-name="$1" --output="$1_%j.out" \
        scripts/hpc/run_shell.slurm "$3"
}
submit gate_lf 2:00:00 "python3 -u scripts/train_impaired_gate.py --data data/processed/stage_b --arch multiscale --img-size 512 --epochs 20 --device cuda --out checkpoints/impaired_gate_multiscale_lf && python3 -u scripts/evaluate_impaired_gate.py --checkpoint checkpoints/impaired_gate_multiscale_lf/impaired_gate_head.pt --data data/processed/stage_b --split test --tune-thresholds --recall-target 0.95 --device cuda"
submit stage_a_lf 5:00:00 "python3 -u scripts/train_stage_a.py --data data/processed/stage_a --taps 8,16,32 --epochs 20 --img-size 640 --device cuda --out checkpoints/stage_a_multiscale_lf && python3 -u scripts/evaluate_stage_a.py --checkpoint checkpoints/stage_a_multiscale_lf/stage_a_head.pt --data data/processed/stage_a --split test --tune-thresholds --by-severity --device cuda"
submit stage_b_lf 3:00:00 "python3 -u scripts/train_stage_b.py --data data/processed/stage_b --arch multiscale --taps 4,8,16,32 --hflip --hidden-dim 32 --fuse-kernel 3 --epochs 40 --loss focal --focal-alpha 0.75 --focal-gamma 2.0 --device cuda --out checkpoints/stage_b_h32_lf"
submit stage_c_lf 6:00:00 "python3 -u scripts/train_stage_c.py --data data/processed/stage_b --arch unet --taps 2,4,8,16,32 --epochs 25 --batch-size 16 --img-size 512 --device cuda --out checkpoints/stage_c_unet_t2_lf"
