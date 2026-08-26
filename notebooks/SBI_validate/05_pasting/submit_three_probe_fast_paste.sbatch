#!/bin/bash
#SBATCH --job-name=sbi3_fast_paste
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-31%8
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/three_probe_mock/logs/fast_paste_%A_%a.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/three_probe_mock/logs/fast_paste_%A_%a.err

set -euo pipefail

REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PYTHON=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
CONFIG=${CONFIG:-${REPO}/notebooks/SBI_validate/three_probe_mock_experiment.yaml}
CATALOG=${CATALOG:-c0000_z0p3_0p5_mmin5e11}
NSIDE=${NSIDE:-512}
NUM_SPLITS=${NUM_SPLITS:-32}

export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR=/tmp/matplotlib

cd "${REPO}"
"${PYTHON}" notebooks/xDESI/paste_abacus_maps.py \
  --config "${CONFIG}" \
  --catalog "${CATALOG}" \
  --split-index "${SLURM_ARRAY_TASK_ID}" \
  --num-splits "${NUM_SPLITS}" \
  --nside "${NSIDE}"
