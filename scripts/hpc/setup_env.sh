#!/bin/bash -l
#
# One-time environment setup for Stage A on NHR@FAU TinyGPU. Run this once,
# from an *interactive* GPU job (not the plain frontend) so GPU support is
# correctly detected when packages are installed --
# see https://doc.nhr.fau.de/environment/python-env/
#
#   ssh tinyx.nhr.fau.de
#   cd /home/woody/<group>/<user>/Farhad_Vaseghi_MA     # your $WORK clone
#   salloc.tinygpu --gres=gpu:1 --time=01:00:00
#   bash scripts/hpc/setup_env.sh
#
# Re-running is safe: `conda config --add` is idempotent, `conda create -y`
# just recreates the env if it already exists.

set -euo pipefail

ENV_NAME="${1:-stage_a}"

# One-time conda init: point conda's package/env storage at $WORK instead
# of the much smaller, backed-up $HOME quota.
if [ ! -f ~/.bash_profile ]; then
  echo "if [ -f ~/.bashrc ]; then . ~/.bashrc; fi" > ~/.bash_profile
fi
module add python
conda config --add pkgs_dirs "$WORK/software/private/conda/pkgs"
conda config --add envs_dirs "$WORK/software/private/conda/envs"

# Some TinyGPU compute nodes have no direct internet access.
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

conda create -y -n "$ENV_NAME" python=3.11
conda activate "$ENV_NAME"

pip install -r requirements.txt

echo ""
echo "Environment '$ENV_NAME' ready. Activate it in batch jobs with:"
echo "  module load python && conda activate $ENV_NAME"
