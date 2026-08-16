#!/usr/bin/env bash
#SBATCH --job-name=bf_gm_bforge_full
#SBATCH --partition=cmbas
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/baryonforge_compare/slurm_logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/baryonforge_compare/slurm_logs/%x.%j.err

set -euo pipefail
umask 027

readonly REPO_ROOT="/mnt/ceph/users/spandey/ltu-godmax/GODMAX"
readonly PYTHON="/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python"
readonly CONFIG="${REPO_ROOT}/notebooks/xDESI/baryonforge_compare/backlight_compare_projection_matched.yaml"
readonly PAINTER="${REPO_ROOT}/notebooks/xDESI/baryonforge_compare/paint_baryonforge.py"
readonly EXPECTED_OUTPUT="${REPO_ROOT}/data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap600_projection_matched/maps/baryonforge_native_nside1024.h5"

export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/bf_gm_matplotlib_${SLURM_JOB_ID}"

mkdir -p "${MPLCONFIGDIR}"
cd "${REPO_ROOT}"

echo "[$(date --iso-8601=seconds)] host=$(hostname) job_id=${SLURM_JOB_ID}"
echo "[$(date --iso-8601=seconds)] python=${PYTHON}"
echo "[$(date --iso-8601=seconds)] config=${CONFIG}"
echo "[$(date --iso-8601=seconds)] expected_output=${EXPECTED_OUTPUT}"
echo "[$(date --iso-8601=seconds)] git_commit=$(git rev-parse --verify HEAD)"
sha256sum "${CONFIG}" "${PAINTER}"
if [[ -e "${EXPECTED_OUTPUT}" ]]; then
  echo "Refusing to overwrite existing BaryonForge output: ${EXPECTED_OUTPUT}" >&2
  exit 64
fi

"${PYTHON}" -u "${PAINTER}" \
  --config "${CONFIG}" \
  --n-jobs 8

test -s "${EXPECTED_OUTPUT}"
echo "[$(date --iso-8601=seconds)] completed output=${EXPECTED_OUTPUT}"
