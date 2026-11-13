#!/bin/bash -l
#
# Runs one Python command on a GPU node, e.g. to render report figures:
#   sbatch -p a100 --gres=gpu:a100:1 --job-name=viz_stage_c scripts/hpc/run_python.slurm \
#       scripts/visualize_stage_c_results.py --checkpoint ... --device cuda
#
#SBATCH --job-name=run_python
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=1:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate stage_a

python3 -u "$@"
