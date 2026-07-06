#!/usr/bin/env bash
#SBATCH --job-name=pz3cap600_preprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
SCRIPT_DIR=${REPO}/notebooks/xDESI/abacus_paste
PYTHON=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
DRIVER=${SCRIPT_DIR}/stage31_pz1_backlight_validation.py
CONFIG=${CONFIG:-${SCRIPT_DIR}/stage31_pz3_cap600.selected.yaml}
MONITOR=${SCRIPT_DIR}/monitor_node_usage.sh

MONITOR_INTERVAL=${MONITOR_INTERVAL:-180}
GPU_SAMPLE_INTERVAL=${GPU_SAMPLE_INTERVAL:-10}
OVERWRITE_CATALOG=${OVERWRITE_CATALOG:-0}

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib_${USER}_${SLURM_JOB_ID}}
mkdir -p "${MPLCONFIGDIR}" "${SCRIPT_DIR}/slurm_logs"
cd "${REPO}"

echo "[preprocess] host=$(hostname)"
echo "[preprocess] config=${CONFIG}"
echo "[preprocess] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "[preprocess] overwrite_catalog=${OVERWRITE_CATALOG}"
echo "[preprocess] monitor_interval=${MONITOR_INTERVAL}s gpu_sample_interval=${GPU_SAMPLE_INTERVAL}s"

MONITOR_PID=""
if [ "${MONITOR_INTERVAL}" -gt 0 ]; then
  MONITOR_LABEL="preprocess_${SLURM_JOB_ID}" \
  MONITOR_INTERVAL="${MONITOR_INTERVAL}" \
  GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL}" \
  "${MONITOR}" &
  MONITOR_PID=$!
  echo "[preprocess] monitor_pid=${MONITOR_PID}"
fi

cleanup_monitor() {
  if [ -n "${MONITOR_PID}" ]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
}
trap cleanup_monitor EXIT

args=()
if [ "${OVERWRITE_CATALOG}" -gt 0 ]; then
  args+=(--overwrite)
fi

srun --ntasks=1 --nodes=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" --cpu-bind=none \
  "${PYTHON}" "${DRIVER}" preprocess \
  --config "${CONFIG}" \
  "${args[@]}"

srun --ntasks=1 --nodes=1 --cpus-per-task=1 --cpu-bind=none \
  "${PYTHON}" "${DRIVER}" catalog-summary \
  --config "${CONFIG}"
