#!/bin/bash
#SBATCH --job-name=sbi3_fast_combine
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/three_probe_mock/logs/fast_combine_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/three_probe_mock/logs/fast_combine_%j.err

set -euo pipefail

REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PYTHON=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
CONFIG=${CONFIG:-${REPO}/notebooks/SBI_validate/three_probe_mock_experiment.yaml}
CATALOG=${CATALOG:-c0000_z0p3_0p5_mmin5e11}
NSIDE=${NSIDE:-512}
NUM_SPLITS=${NUM_SPLITS:-32}

export JAX_ENABLE_X64=True
export MPLCONFIGDIR=/tmp/matplotlib

cd "${REPO}"
"${PYTHON}" notebooks/xDESI/combine_abacus_map_splits.py \
  --config "${CONFIG}" \
  --catalog "${CATALOG}" \
  --num-splits "${NUM_SPLITS}" \
  --nside "${NSIDE}"
