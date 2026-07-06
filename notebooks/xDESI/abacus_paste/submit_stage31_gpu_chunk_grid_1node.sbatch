#!/usr/bin/env bash
#SBATCH --job-name=stage31_chunk_grid
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --mem=0
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
SCRIPT_DIR=${REPO}/notebooks/xDESI/abacus_paste
PYTHON=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
DRIVER=${SCRIPT_DIR}/stage31_pz1_backlight_validation.py
CONFIG=${CONFIG:-${SCRIPT_DIR}/stage31_pz3_cap600.selected.yaml}

NSIDE1024_NHALOS_LIST=${NSIDE1024_NHALOS_LIST:-1000000 1250000 1500000 2000000}
NSIDE2048_NHALOS_LIST=${NSIDE2048_NHALOS_LIST:-250000 500000 750000 1000000}
SAMPLE_MODE_LIST=${SAMPLE_MODE_LIST:-random largest-paint largest-mass lowest-z}
PIXEL_WORKERS=${PIXEL_WORKERS:-16}
PIXEL_POOL_CHUNKSIZE=${PIXEL_POOL_CHUNKSIZE:-32}
PIXEL_BACKEND=${PIXEL_BACKEND:-healpy}
QUERY_DISC_BUFFER_SAFETY_FACTOR=${QUERY_DISC_BUFFER_SAFETY_FACTOR:-2.0}
GALAXY_POPULATION_BACKEND=${GALAXY_POPULATION_BACKEND:-compact}
GALAXY_COMPACT_MAX_SATELLITE_GROUPS=${GALAXY_COMPACT_MAX_SATELLITE_GROUPS:-8}
INCLUDE_GALAXIES=${INCLUDE_GALAXIES:-1}

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib_${USER}_${SLURM_JOB_ID}}
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export PASTE_JAX_PLATFORMS=cuda
export JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p "${MPLCONFIGDIR}" "${SCRIPT_DIR}/slurm_logs"
cd "${REPO}"

echo "[chunk-grid] host=$(hostname)"
echo "[chunk-grid] config=${CONFIG}"
echo "[chunk-grid] nside1024=${NSIDE1024_NHALOS_LIST}"
echo "[chunk-grid] nside2048=${NSIDE2048_NHALOS_LIST}"
echo "[chunk-grid] sample_modes=${SAMPLE_MODE_LIST}"
echo "[chunk-grid] hod_backend=${GALAXY_POPULATION_BACKEND}"
echo "[chunk-grid] compact_max_satellite_groups=${GALAXY_COMPACT_MAX_SATELLITE_GROUPS}"

run_case() {
  local NSIDE=$1
  local NHALOS=$2
  local SAMPLE_MODE=$3
  local BACKEND_TAG=${PIXEL_BACKEND//-/_}
  local HOD_TAG=${GALAXY_POPULATION_BACKEND//-/_}
  local SAMPLE_TAG=${SAMPLE_MODE//-/_}
  local OUT="${REPO}/data/xDESI/processed/abacus_backlight/stage31_pz3_cap600/measurements/gpu_chunk_grid_${BACKEND_TAG}_${HOD_TAG}_${SAMPLE_TAG}_nside${NSIDE}_halos${NHALOS}_job${SLURM_JOB_ID}.json"
  local GALAXY_ARG=--no-include-galaxies
  if [ "${INCLUDE_GALAXIES}" -gt 0 ]; then
    GALAXY_ARG=--include-galaxies
  fi
  echo "[chunk-grid] nside=${NSIDE} n_halos=${NHALOS} sample=${SAMPLE_MODE} output=${OUT}"
  if ! srun --ntasks=1 --nodes=1 --gres=gpu:1 --cpus-per-task="${PIXEL_WORKERS}" --cpu-bind=none \
    "${PYTHON}" "${DRIVER}" benchmark-gpu-chunk \
    --config "${CONFIG}" \
    --nside "${NSIDE}" \
    --n-halos "${NHALOS}" \
    --sample-mode "${SAMPLE_MODE}" \
    --pixel-workers "${PIXEL_WORKERS}" \
    --pixel-pool-chunksize "${PIXEL_POOL_CHUNKSIZE}" \
    --pixel-backend "${PIXEL_BACKEND}" \
    --query-disc-buffer-safety-factor "${QUERY_DISC_BUFFER_SAFETY_FACTOR}" \
    --single-pixel-angle-factor 0.5 \
    "${GALAXY_ARG}" \
    --galaxy-population-backend "${GALAXY_POPULATION_BACKEND}" \
    --galaxy-compact-max-satellite-groups "${GALAXY_COMPACT_MAX_SATELLITE_GROUPS}" \
    --output "${OUT}" \
    --quiet; then
    echo "[chunk-grid] FAILED nside=${NSIDE} n_halos=${NHALOS} sample=${SAMPLE_MODE}" >&2
  fi
}

for SAMPLE_MODE in ${SAMPLE_MODE_LIST}; do
  for NHALOS in ${NSIDE1024_NHALOS_LIST}; do
    run_case 1024 "${NHALOS}" "${SAMPLE_MODE}"
  done
  for NHALOS in ${NSIDE2048_NHALOS_LIST}; do
    run_case 2048 "${NHALOS}" "${SAMPLE_MODE}"
  done
done
