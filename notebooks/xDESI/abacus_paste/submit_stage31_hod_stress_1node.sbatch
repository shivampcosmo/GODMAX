#!/usr/bin/env bash
#SBATCH --job-name=stage31_hod_stress
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --mem=0
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
SCRIPT_DIR=${REPO}/notebooks/xDESI/abacus_paste
PYTHON=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
DRIVER=${SCRIPT_DIR}/stage31_pz1_backlight_validation.py
CONFIG=${CONFIG:-${SCRIPT_DIR}/stage31_pz3_cap600.selected.yaml}

NSIDE_LIST=${NSIDE_LIST:-1024,2048}
SAMPLE_MODES=${SAMPLE_MODES:-random,largest-paint,largest-mass,lowest-z,highest-hod}
PIXEL_SAMPLE_HALOS=${PIXEL_SAMPLE_HALOS:-50000}
PIXEL_WORKERS=${PIXEL_WORKERS:-16}
PIXEL_POOL_CHUNKSIZE=${PIXEL_POOL_CHUNKSIZE:-32}
PIXEL_START_METHOD=${PIXEL_START_METHOD:-forkserver}
HOD_PLATFORM=${HOD_PLATFORM:-cuda}
HOD_CHUNK_SIZE=${HOD_CHUNK_SIZE:-200000}
INCLUDE_PIXEL_WORK=${INCLUDE_PIXEL_WORK:-1}
Z_BINS=${Z_BINS:-8}

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib_${USER}_${SLURM_JOB_ID}}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PASTE_JAX_PLATFORMS=${HOD_PLATFORM}
export JAX_PLATFORMS=${HOD_PLATFORM}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p "${MPLCONFIGDIR}" "${SCRIPT_DIR}/slurm_logs"
cd "${REPO}"

TAG=$(basename "${CONFIG}" .yaml)
OUT="${REPO}/data/xDESI/processed/abacus_backlight/stage31_pz3_cap600/measurements/hod_stress_${TAG}_job${SLURM_JOB_ID}.json"
PIXEL_ARG=--no-include-pixel-work
if [ "${INCLUDE_PIXEL_WORK}" -gt 0 ]; then
  PIXEL_ARG=--include-pixel-work
fi

echo "[hod-stress] host=$(hostname)"
echo "[hod-stress] config=${CONFIG}"
echo "[hod-stress] nside_list=${NSIDE_LIST}"
echo "[hod-stress] sample_modes=${SAMPLE_MODES}"
echo "[hod-stress] pixel_sample_halos=${PIXEL_SAMPLE_HALOS}"
echo "[hod-stress] pixel_start_method=${PIXEL_START_METHOD}"
echo "[hod-stress] hod_platform=${HOD_PLATFORM}"
echo "[hod-stress] output=${OUT}"

srun --ntasks=1 --nodes=1 --gres=gpu:1 --cpus-per-task="${PIXEL_WORKERS}" --cpu-bind=none \
  "${PYTHON}" "${DRIVER}" diagnose-hod-stress \
  --config "${CONFIG}" \
  --nside-list "${NSIDE_LIST}" \
  --sample-modes "${SAMPLE_MODES}" \
  --pixel-sample-halos "${PIXEL_SAMPLE_HALOS}" \
  --pixel-workers "${PIXEL_WORKERS}" \
  --pixel-pool-chunksize "${PIXEL_POOL_CHUNKSIZE}" \
  --pixel-start-method "${PIXEL_START_METHOD}" \
  --hod-platform "${HOD_PLATFORM}" \
  --hod-chunk-size "${HOD_CHUNK_SIZE}" \
  --z-bins "${Z_BINS}" \
  ${PIXEL_ARG} \
  --output "${OUT}" \
  --quiet
