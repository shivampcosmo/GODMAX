#!/usr/bin/env bash
#SBATCH --job-name=pz3_pixel_scaling
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --mem=0
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
SCRIPT_DIR=${REPO}/notebooks/xDESI/abacus_paste
PYTHON=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
DRIVER=${SCRIPT_DIR}/stage31_pz1_backlight_validation.py
CONFIG=${SCRIPT_DIR}/stage31_pz3_cap600.selected.yaml

NSIDE_LIST=${NSIDE_LIST:-512 1024}
SAMPLE_SIZES=${SAMPLE_SIZES:-50000,200000}
WORKERS=${WORKERS:-1,8,16,32,64}
POOL_CHUNKSIZES=${POOL_CHUNKSIZES:-0,32,128,512}
SHORTCUT_FACTORS=${SHORTCUT_FACTORS:-0 0.5}
PIXEL_BACKENDS=${PIXEL_BACKENDS:-healpy healpy-buff}
QUERY_DISC_BUFFER_SAFETY_FACTOR=${QUERY_DISC_BUFFER_SAFETY_FACTOR:-2.0}

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib_${USER}_${SLURM_JOB_ID}}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p "${MPLCONFIGDIR}" "${SCRIPT_DIR}/slurm_logs"
cd "${REPO}"

echo "[pixel-scaling] host=$(hostname)"
echo "[pixel-scaling] config=${CONFIG}"
echo "[pixel-scaling] sample_sizes=${SAMPLE_SIZES}"
echo "[pixel-scaling] workers=${WORKERS}"
echo "[pixel-scaling] pool_chunksizes=${POOL_CHUNKSIZES}"
echo "[pixel-scaling] shortcut_factors=${SHORTCUT_FACTORS}"
echo "[pixel-scaling] pixel_backends=${PIXEL_BACKENDS}"
echo "[pixel-scaling] query_disc_buffer_safety_factor=${QUERY_DISC_BUFFER_SAFETY_FACTOR}"

for BACKEND in ${PIXEL_BACKENDS}; do
  BACKEND_TAG=${BACKEND//-/_}
  for NSIDE in ${NSIDE_LIST}; do
    for SHORTCUT in ${SHORTCUT_FACTORS}; do
      OUT="${REPO}/data/xDESI/processed/abacus_backlight/stage31_pz3_cap600/measurements/pixel_work_scaling_${BACKEND_TAG}_nside${NSIDE}_shortcut${SHORTCUT//./p}_job${SLURM_JOB_ID}.json"
      echo "[pixel-scaling] backend=${BACKEND} nside=${NSIDE} shortcut=${SHORTCUT} output=${OUT}"
      srun --ntasks=1 --nodes=1 --cpus-per-task="${SLURM_CPUS_ON_NODE:-$(nproc)}" --cpu-bind=none \
        "${PYTHON}" "${DRIVER}" benchmark-pixel-work \
        --config "${CONFIG}" \
        --nside "${NSIDE}" \
        --sample-sizes "${SAMPLE_SIZES}" \
        --workers "${WORKERS}" \
        --pool-chunksizes "${POOL_CHUNKSIZES}" \
        --single-pixel-angle-factor "${SHORTCUT}" \
        --pixel-backend "${BACKEND}" \
        --query-disc-buffer-safety-factor "${QUERY_DISC_BUFFER_SAFETY_FACTOR}" \
        --output "${OUT}" \
        --quiet
    done
  done
done
