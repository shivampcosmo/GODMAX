#!/usr/bin/env bash
#SBATCH --job-name=pz1cap600_paste
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --array=0-3%4
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%A_%a.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%A_%a.err

set -euo pipefail

REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
SCRIPT_DIR=${REPO}/notebooks/xDESI/abacus_paste
PYTHON=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
DRIVER=${SCRIPT_DIR}/stage31_pz1_backlight_validation.py
CONFIG=${SCRIPT_DIR}/stage31_pz1_cap600.selected.yaml
MONITOR=${SCRIPT_DIR}/monitor_node_usage.sh

NSIDE=${NSIDE:-1024}
NUM_SPLITS=${NUM_SPLITS:-4}
SPLIT_INDEX=${SLURM_ARRAY_TASK_ID}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-180}
GPU_SAMPLE_INTERVAL=${GPU_SAMPLE_INTERVAL:-10}
JAX_BACKEND=${JAX_BACKEND:-cuda}
JAX_PREFLIGHT=${JAX_PREFLIGHT:-1}

# Slurm reports full node CPU count for exclusive allocations.  Fall back to
# nproc for clusters that do not set SLURM_CPUS_ON_NODE.
PIXEL_WORKERS=${PIXEL_WORKERS:-${SLURM_CPUS_ON_NODE:-$(nproc)}}
PIXEL_START_METHOD=${PIXEL_START_METHOD:-fork}

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib_${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export PASTE_JAX_PLATFORMS="${JAX_BACKEND}"
export JAX_PLATFORMS="${JAX_BACKEND}"

# Keep CPU libraries from oversubscribing the query_disc multiprocessing pool.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p "${MPLCONFIGDIR}" "${REPO}/notebooks/xDESI/abacus_paste/slurm_logs"
cd "${REPO}"

echo "[submit] host=$(hostname)"
echo "[submit] job=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID}"
echo "[submit] nside=${NSIDE} split=${SPLIT_INDEX}/${NUM_SPLITS}"
echo "[submit] SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-unset} PIXEL_WORKERS=${PIXEL_WORKERS}"
echo "[submit] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[submit] XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION}"
echo "[submit] JAX_PLATFORMS=${JAX_PLATFORMS}"
echo "[submit] monitor_interval=${MONITOR_INTERVAL}s gpu_sample_interval=${GPU_SAMPLE_INTERVAL}s"

MONITOR_PID=""
if [ "${MONITOR_INTERVAL}" -gt 0 ]; then
  MONITOR_LABEL="paste_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
  MONITOR_INTERVAL="${MONITOR_INTERVAL}" \
  GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL}" \
  "${MONITOR}" &
  MONITOR_PID=$!
  echo "[submit] monitor_pid=${MONITOR_PID}"
fi

cleanup_monitor() {
  if [ -n "${MONITOR_PID}" ]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
}
trap cleanup_monitor EXIT

if [ "${JAX_PREFLIGHT}" -gt 0 ]; then
  echo "[submit] jax preflight start"
  srun --ntasks=1 --nodes=1 --cpus-per-task=1 --cpu-bind=none "${PYTHON}" -c 'import jax, jaxlib; print(f"[jax-preflight] jax={jax.__version__} jaxlib={jaxlib.__version__}"); print(f"[jax-preflight] backend={jax.default_backend()} devices={jax.devices()}")'
  echo "[submit] jax preflight done"
fi

srun --ntasks=1 --nodes=1 --cpus-per-task="${PIXEL_WORKERS}" --cpu-bind=none "${PYTHON}" "${DRIVER}" paste-split \
  --config "${CONFIG}" \
  --nside "${NSIDE}" \
  --split-index "${SPLIT_INDEX}" \
  --num-splits "${NUM_SPLITS}" \
  --pixel-workers "${PIXEL_WORKERS}" \
  --pixel-start-method "${PIXEL_START_METHOD}" \
  --overwrite \
  --verbose
