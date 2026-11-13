#!/bin/bash
# Session 27: train Stages A, B and C on the visible-change dataset, one job each.
#   DATA=data/processed/visible PART=a100 bash scripts/hpc/train_visible.sh [dependency job id]
#   PART=v100 ... uses the stage_a_v100 env; PART=work runs on rtx3080/rtx2080ti.
# SUFFIX is appended to the checkpoint dirs and job names (for twin submissions).
set -e
DEP=${1:+--dependency=afterok:$1}
DATA=${DATA:-data/processed/visible}
PART=${PART:-a100}
SUFFIX=${SUFFIX:-}
ENV=stage_a
GRES=gpu:$PART:1
[ "$PART" = v100 ] && ENV=stage_a_v100
[ "$PART" = work ] && GRES=gpu:1
submit() {  # stage, time
    sbatch.tinygpu -p "$PART" --gres=$GRES --time="$2" $DEP --export=NONE,CONDA_ENV=$ENV \
        --job-name="visible_$1$SUFFIX" --output="visible_${1}${SUFFIX}_%j.out" \
        scripts/hpc/run_shell.slurm "python3 -u scripts/train_visible.py --stage $1 --data $DATA --device cuda --out checkpoints/visible_$1$SUFFIX"
}
submit a 3:00:00
submit b 4:00:00
submit c 8:00:00
