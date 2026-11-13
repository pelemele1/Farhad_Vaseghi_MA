#!/bin/bash
# Session 27: tune the gate and map offsets on val, evaluate the pipeline on test, draw the figures.
#   DATA=data/processed/visible PART=a100 A=checkpoints/visible_a B=checkpoints/visible_b \
#   C=checkpoints/visible_c LOGS=a.out,b.out,c.out bash scripts/hpc/evaluate_visible.sh [dependency job id]
set -e
DEP=${1:+--dependency=afterok:$1}
DATA=${DATA:-data/processed/visible}
PART=${PART:-a100}
A=${A:-checkpoints/visible_a}
B=${B:-checkpoints/visible_b}
C=${C:-checkpoints/visible_c}
ENV=stage_a
GRES=gpu:$PART:1
[ "$PART" = v100 ] && ENV=stage_a_v100
[ "$PART" = work ] && GRES=gpu:1
CK="--a $A/stage_a_head.pt --b $B/stage_b_head.pt --c $C/stage_c_head.pt"
FIGS=""
[ -n "$LOGS" ] && FIGS="--logs $LOGS"
sbatch.tinygpu -p "$PART" --gres=$GRES --time=3:00:00 $DEP --export=NONE,CONDA_ENV=$ENV \
    --job-name=eval_visible --output="eval_visible_%j.out" scripts/hpc/run_shell.slurm \
    "python3 -u scripts/evaluate_visible.py --data $DATA $CK --device cuda --config checkpoints/visible_pipeline.json --out-json results/visible_eval.json && python3 -u scripts/visualize_visible.py --data $DATA $CK --config checkpoints/visible_pipeline.json --results results/visible_eval.json $FIGS --device cuda --out-dir docs/images"
