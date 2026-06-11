#!/usr/bin/env bash
#SBATCH --job-name=pz3_jax_healpy
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=0
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
SCRIPT_DIR=${REPO}/notebooks/xDESI/abacus_paste
CONDA=${CONDA:-/mnt/home/spandey/miniconda3/bin/conda}
CONDA_ENV=${CONDA_ENV:-godmax-jaxhealpy}
DRIVER=${SCRIPT_DIR}/stage31_pz1_backlight_validation.py
CONFIG=${CONFIG:-${SCRIPT_DIR}/stage31_pz3_cap600.selected.yaml}

NSIDE=${NSIDE:-1024}
NHALOS=${NHALOS:-2000}
QUERY_DISC_COUNT=${QUERY_DISC_COUNT:-200}
INCLUDE_JAX_QUERY_DISC=${INCLUDE_JAX_QUERY_DISC:-1}
INCLUDE_JAX_QUERY_DISC_LOOP=${INCLUDE_JAX_QUERY_DISC_LOOP:-0}
JAX_HEALPY_PATH=${JAX_HEALPY_PATH:-/mnt/ceph/users/spandey/ltu-godmax/jax-healpy}

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib_${USER}_${SLURM_JOB_ID}}
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.25}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p "${MPLCONFIGDIR}" "${SCRIPT_DIR}/slurm_logs"
cd "${REPO}"

OUT="${REPO}/data/xDESI/processed/abacus_backlight/stage31_pz3_cap600/measurements/jax_healpy_function_gpu_nside${NSIDE}_halos${NHALOS}_qdisc${QUERY_DISC_COUNT}_job${SLURM_JOB_ID}.json"

echo "[jax-healpy] host=$(hostname)"
echo "[jax-healpy] conda_env=${CONDA_ENV}"
echo "[jax-healpy] config=${CONFIG}"
echo "[jax-healpy] nside=${NSIDE} n_halos=${NHALOS} query_disc_count=${QUERY_DISC_COUNT}"
echo "[jax-healpy] output=${OUT}"

JAX_QUERY_ARG=()
if [ "${INCLUDE_JAX_QUERY_DISC}" -gt 0 ]; then
  JAX_QUERY_ARG=(--include-jax-query-disc)
fi
JAX_QUERY_LOOP_ARG=()
if [ "${INCLUDE_JAX_QUERY_DISC_LOOP}" -gt 0 ]; then
  JAX_QUERY_LOOP_ARG=(--include-jax-query-disc-loop)
fi

srun --ntasks=1 --nodes=1 --gres=gpu:1 --cpus-per-task="${SLURM_CPUS_ON_NODE:-8}" --cpu-bind=none \
  "${CONDA}" run -n "${CONDA_ENV}" python "${DRIVER}" benchmark-healpix-functions \
  --config "${CONFIG}" \
  --nside "${NSIDE}" \
  --n-halos "${NHALOS}" \
  --query-disc-count "${QUERY_DISC_COUNT}" \
  --run-jax-healpy \
  --jax-healpy-path "${JAX_HEALPY_PATH}" \
  --jax-device cuda \
  "${JAX_QUERY_ARG[@]}" \
  "${JAX_QUERY_LOOP_ARG[@]}" \
  --output "${OUT}"
