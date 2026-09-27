#!/bin/bash
# Stage B overfitting, part 2 (Session 25): the flip alone shrank but did not close the
# train/val gap, so the head capacity is reduced (fewer channels, 1x1 instead of 3x3 fusion),
# all with --hflip, trained and scored on the VAL split. a100 only: twin copies that start in
# the same scheduler cycle would write the same checkpoint directory.
#   bash scripts/hpc/stage_b_small_head.sh   (from $WORK/Farhad_Vaseghi_MA)
set -e
OUT=checkpoints/study
mkdir -p study_logs
submit() {  # name, hidden_dim, fuse_kernel
    local name=$1
    local cmd="python3 -u scripts/train_stage_b.py --data data/processed/stage_b --arch multiscale --taps 4,8,16,32 --hflip --hidden-dim $2 --fuse-kernel $3 --epochs 40 --loss focal --focal-alpha 0.75 --focal-gamma 2.0 --device cuda --out $OUT/$name && python3 -u scripts/evaluate_stage_b.py --checkpoint $OUT/$name/stage_b_head.pt --data data/processed/stage_b --split val --tune-thresholds --by-severity --device cuda"
    sbatch.tinygpu -p a100 --gres=gpu:a100:1 --time=3:00:00 --job-name="$name" --output="study_logs/${name}_%j.out" scripts/hpc/run_shell.slurm "$cmd"
}
submit b_small_h32k3 32 3
submit b_small_h32k1 32 1
submit b_small_h16k3 16 3
submit b_small_h16k1 16 1
submit b_small_h8k1 8 1
