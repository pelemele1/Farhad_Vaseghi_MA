#!/bin/bash -l
#
# Runs a shell command line on a GPU node, e.g. train then evaluate in one job:
#   sbatch -p a100 --gres=gpu:a100:1 scripts/hpc/run_shell.slurm "python3 -u a.py && python3 -u b.py"
#   v100: sbatch -p v100 --gres=gpu:v100:1 --export=NONE,CONDA_ENV=stage_a_v100 scripts/hpc/run_shell.slurm "..."
#
#SBATCH --job-name=run_shell
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=1:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate "${CONDA_ENV:-stage_a}"   # stage_a_v100 on the v100 partition (torch build with sm_70)

bash -c "$1"
