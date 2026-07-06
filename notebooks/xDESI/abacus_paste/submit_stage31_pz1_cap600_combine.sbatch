#!/usr/bin/env bash
#SBATCH --job-name=pz1cap600_combine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
SCRIPT_DIR=${REPO}/notebooks/xDESI/abacus_paste
PYTHON=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
DRIVER=${SCRIPT_DIR}/stage31_pz1_backlight_validation.py
CONFIG=${SCRIPT_DIR}/stage31_pz1_cap600.selected.yaml
MONITOR=${SCRIPT_DIR}/monitor_node_usage.sh

NSIDE=${NSIDE:-1024}
NUM_SPLITS=${NUM_SPLITS:-4}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-180}
GPU_SAMPLE_INTERVAL=${GPU_SAMPLE_INTERVAL:-10}

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib_${USER}_${SLURM_JOB_ID}}
mkdir -p "${MPLCONFIGDIR}" "${REPO}/notebooks/xDESI/abacus_paste/slurm_logs"
cd "${REPO}"

echo "[combine] host=$(hostname)"
echo "[combine] nside=${NSIDE} num_splits=${NUM_SPLITS}"
echo "[combine] monitor_interval=${MONITOR_INTERVAL}s gpu_sample_interval=${GPU_SAMPLE_INTERVAL}s"

MONITOR_PID=""
if [ "${MONITOR_INTERVAL}" -gt 0 ]; then
  MONITOR_LABEL="combine_${SLURM_JOB_ID}" \
  MONITOR_INTERVAL="${MONITOR_INTERVAL}" \
  GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL}" \
  "${MONITOR}" &
  MONITOR_PID=$!
  echo "[combine] monitor_pid=${MONITOR_PID}"
fi

cleanup_monitor() {
  if [ -n "${MONITOR_PID}" ]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
}
trap cleanup_monitor EXIT

srun --ntasks=1 --nodes=1 "${PYTHON}" "${DRIVER}" combine-maps \
  --config "${CONFIG}" \
  --nside "${NSIDE}" \
  --num-splits "${NUM_SPLITS}" \
  --overwrite

srun --ntasks=1 --nodes=1 "${PYTHON}" "${DRIVER}" plot-healpix-maps \
  --config "${CONFIG}" \
  --nside "${NSIDE}"
