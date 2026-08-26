#!/bin/bash
#SBATCH --job-name=sbi-noise-draw
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-11%6
#SBATCH --output=data/SBI_validate/three_probe_mock/logs/noise_draw_%A_%a.out
#SBATCH --error=data/SBI_validate/three_probe_mock/logs/noise_draw_%A_%a.err

set -euo pipefail
REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PYTHON=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
ROOT=${REPO}/data/SBI_validate/three_probe_mock/validation/noisy_nside1024_ell2048
INDEX=${SLURM_ARRAY_TASK_ID}
OUTPUT=$(printf '%s/realizations/noise_realization_%03d.h5' "${ROOT}" "${INDEX}")
cd "${REPO}"
mkdir -p "${ROOT}/realizations" "${REPO}/data/SBI_validate/three_probe_mock/logs"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"${PYTHON}" notebooks/SBI_validate/common/three_probe_noise_contract.py realize \
  --contract "${ROOT}/noise_contract.h5" \
  --realization "${INDEX}" \
  --output "${OUTPUT}"

