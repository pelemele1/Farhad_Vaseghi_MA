#!/bin/bash -l
#
# Stage C evaluation on a GPU (tuned thresholds, per-severity breakdown,
# optional gate). Usage:
#   sbatch -p a100 --gres=gpu:a100:1 --export=NONE,CKPT=checkpoints/stage_c_unet/stage_c_head.pt,GATE_T=0.140 \
#       scripts/hpc/evaluate_stage_c.slurm
# DATA defaults to data/processed/stage_b_scratch15; omit GATE_T for no gate; GATE_CKPT
# defaults to the multi-scale gate.
#
#SBATCH --job-name=eval_stage_c
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=1:00:00
#SBATCH --output=%x_%j.out

unset SLURM_EXPORT_ENV

module load python
conda activate stage_a

DATA=${DATA:-data/processed/stage_b_scratch15}
GATE_ARGS=()
if [ -n "$GATE_T" ]; then
    GATE_ARGS=(--gate-checkpoint "${GATE_CKPT:-checkpoints/impaired_gate_multiscale/impaired_gate_head.pt}" --gate-threshold "$GATE_T")
fi

echo "checkpoint=$CKPT data=$DATA gate_threshold=${GATE_T:-none}"
python3 -u scripts/evaluate_stage_c.py --checkpoint "$CKPT" --data "$DATA" --split test \
    --tune-thresholds --by-severity --batch-size 16 --device cuda "${GATE_ARGS[@]}"
