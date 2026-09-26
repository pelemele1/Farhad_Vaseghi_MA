#!/bin/bash
# Session 23 layer study: which backbone layers (strides) help each stage.
# Every configuration trains with identical settings, then is evaluated on
# the VAL split (selection happens on val; test stays untouched until the
# winner is fixed). Run from $WORK/Farhad_Vaseghi_MA on the TinyGPU login node:
#   bash scripts/hpc/layer_study.sh
set -e
# Each configuration is ONE job (train, then evaluate on val), submitted twice --
# on a100 and on the rtx2080ti/rtx3080 "work" pool -- and whichever copy starts
# first wins; the other copy is cancelled (by hand or by a watcher) once one runs.
OUT=checkpoints/study
mkdir -p study_logs

submit() {  # name, time, "train cmd", "eval cmd"
    local name=$1 time=$2 cmd="python3 -u $3 && python3 -u $4"
    for part in a100 work; do
        local gres=gpu:1
        [ "$part" = a100 ] && gres=gpu:a100:1
        sbatch.tinygpu -p "$part" --gres="$gres" --time="$time" --job-name="$name"             --output="study_logs/${name}_%j.out" scripts/hpc/run_shell.slurm "$cmd" >/dev/null
    done
    echo "$name submitted (a100 + work)"
}

evaluate_only() {  # name, "eval cmd" -- the existing 4,8,16,32 models, scored on val too
    local name=$1 cmd="python3 -u $2"
    for part in a100 work; do
        local gres=gpu:1
        [ "$part" = a100 ] && gres=gpu:a100:1
        sbatch.tinygpu -p "$part" --gres="$gres" --time=1:00:00 --job-name="$name"             --output="study_logs/${name}_%j.out" scripts/hpc/run_shell.slurm "$cmd" >/dev/null
    done
    echo "$name submitted (a100 + work)"
}

evaluate_only b_t4-8-16-32 "scripts/evaluate_stage_b.py --checkpoint checkpoints/stage_b_multiscale/stage_b_head.pt     --data data/processed/stage_b --split val --tune-thresholds --by-severity --device cuda"
evaluate_only c_t4-8-16-32 "scripts/evaluate_stage_c.py --checkpoint checkpoints/stage_c_unet/stage_c_head.pt     --data data/processed/stage_b --split val --tune-thresholds --by-severity --batch-size 16 --device cuda"
evaluate_only g_t4-8-16-32 "scripts/evaluate_impaired_gate.py     --checkpoint checkpoints/impaired_gate_multiscale/impaired_gate_head.pt     --data data/processed/stage_b --split val --tune-thresholds --recall-target 0.95 --device cuda"

for taps in 32 16,32 8,16,32 4,8,16,32 2,4,8,16,32; do
    n=a_t${taps//,/-}
    submit "$n" 3:00:00 "scripts/train_stage_a.py --data data/processed/stage_a --taps $taps         --epochs 20 --img-size 640 --device cuda --out $OUT/$n"         "scripts/evaluate_stage_a.py --checkpoint $OUT/$n/stage_a_head.pt --data data/processed/stage_a         --split val --tune-thresholds --by-severity --device cuda"
done

for taps in 32 16,32 8,16,32 2,4,8,16,32; do
    n=b_t${taps//,/-}
    submit "$n" 3:00:00 "scripts/train_stage_b.py --data data/processed/stage_b --arch multiscale --taps $taps         --epochs 40 --loss focal --focal-alpha 0.75 --focal-gamma 2.0 --device cuda --out $OUT/$n"         "scripts/evaluate_stage_b.py --checkpoint $OUT/$n/stage_b_head.pt --data data/processed/stage_b         --split val --tune-thresholds --by-severity --device cuda"
done

for taps in 16,32 8,16,32 2,4,8,16,32; do
    n=c_t${taps//,/-}
    submit "$n" 5:00:00 "scripts/train_stage_c.py --data data/processed/stage_b --arch unet --taps $taps         --epochs 25 --device cuda --out $OUT/$n"         "scripts/evaluate_stage_c.py --checkpoint $OUT/$n/stage_c_head.pt --data data/processed/stage_b         --split val --tune-thresholds --by-severity --batch-size 16 --device cuda"
done

for taps in 32 8,16,32 2,4,8,16,32; do
    n=g_t${taps//,/-}
    submit "$n" 3:00:00 "scripts/train_impaired_gate.py --data data/processed/stage_b --arch multiscale         --taps $taps --img-size 512 --epochs 20 --device cuda --out $OUT/$n"         "scripts/evaluate_impaired_gate.py --checkpoint $OUT/$n/impaired_gate_head.pt         --data data/processed/stage_b --split val --tune-thresholds --recall-target 0.95 --device cuda"
done
