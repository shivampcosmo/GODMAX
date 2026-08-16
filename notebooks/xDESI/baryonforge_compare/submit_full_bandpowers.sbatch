#!/usr/bin/env bash
#SBATCH --job-name=bf_gm_bandpowers
#SBATCH --partition=cmbas
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/baryonforge_compare/slurm_logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/baryonforge_compare/slurm_logs/%x.%j.err

set -euo pipefail
umask 027

readonly REPO_ROOT="/mnt/ceph/users/spandey/ltu-godmax/GODMAX"
readonly PYTHON="/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python"
readonly CONFIG="${REPO_ROOT}/notebooks/xDESI/baryonforge_compare/backlight_compare_projection_matched.yaml"
readonly MEASURER="${REPO_ROOT}/notebooks/xDESI/baryonforge_compare/measure_statistics.py"
readonly GODMAX_MAP="${REPO_ROOT}/data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap600_projection_matched/maps/godmax_projection_matched_mgt13_pz3_cap600/abacus_pasted_maps_buffered_mgt13_nside1024_split000of001.h5"
readonly BARYONFORGE_MAP="${REPO_ROOT}/data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap600_projection_matched/maps/baryonforge_native_nside1024.h5"
readonly OUTPUT="${REPO_ROOT}/data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap600_projection_matched/measurements/common_mask_statistics.h5"

export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/bf_gm_matplotlib_${SLURM_JOB_ID}"

mkdir -p "${MPLCONFIGDIR}"
cd "${REPO_ROOT}"

echo "[$(date --iso-8601=seconds)] host=$(hostname) job_id=${SLURM_JOB_ID}"
echo "[$(date --iso-8601=seconds)] python=${PYTHON}"
echo "[$(date --iso-8601=seconds)] config=${CONFIG}"
echo "[$(date --iso-8601=seconds)] godmax_map=${GODMAX_MAP}"
echo "[$(date --iso-8601=seconds)] baryonforge_map=${BARYONFORGE_MAP}"
echo "[$(date --iso-8601=seconds)] output=${OUTPUT}"
echo "[$(date --iso-8601=seconds)] git_commit=$(git rev-parse --verify HEAD)"
if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing to overwrite existing statistics output: ${OUTPUT}" >&2
  exit 64
fi
sha256sum "${CONFIG}" "${MEASURER}" "${GODMAX_MAP}" "${BARYONFORGE_MAP}"

"${PYTHON}" -u "${MEASURER}" \
  --config "${CONFIG}" \
  --godmax-maps "${GODMAX_MAP}" \
  --baryonforge-maps "${BARYONFORGE_MAP}" \
  --output "${OUTPUT}"

test -s "${OUTPUT}"
echo "[$(date --iso-8601=seconds)] completed output=${OUTPUT}"
