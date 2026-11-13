#!/bin/bash
# Stage B overfitting study: validation loss bottoms out around epoch 23 of 40
# while train loss keeps falling (800 distinct training scenes, no augmentation
# or regularization). Each run is the canonical 4,8,16,32 multiscale setup plus
# one regularizer (or all three), trained and scored on the VAL split -- same
# twin a100/work submission as layer_study.sh. Run from $WORK/Farhad_Vaseghi_MA:
#   bash scripts/hpc/stage_b_regularization.sh
set -e
OUT=checkpoints/study
mkdir -p study_logs

submit() {  # name, extra train flags
    local name=$1
    local cmd="python3 -u scripts/train_stage_b.py --data data/processed/stage_b --arch multiscale --taps 4,8,16,32 \
--epochs 40 --loss focal --focal-alpha 0.75 --focal-gamma 2.0 $2 --device cuda --out $OUT/$name \
&& python3 -u scripts/evaluate_stage_b.py --checkpoint $OUT/$name/stage_b_head.pt --data data/processed/stage_b \
--split val --tune-thresholds --by-severity --device cuda"
    for part in a100 work; do
        local gres=gpu:1
        [ "$part" = a100 ] && gres=gpu:a100:1
        sbatch.tinygpu -p "$part" --gres="$gres" --time=3:00:00 --job-name="$name" \
            --output="study_logs/${name}_%j.out" scripts/hpc/run_shell.slurm "$cmd" >/dev/null
    done
    echo "$name submitted (a100 + work)"
}

submit b_reg_flip "--hflip"
submit b_reg_wd "--weight-decay 0.05"
submit b_reg_drop "--dropout 0.2"
submit b_reg_all "--hflip --weight-decay 0.05 --dropout 0.2"
submit b_reg_flip_wd "--hflip --weight-decay 0.05"
