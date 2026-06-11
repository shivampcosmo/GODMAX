#!/usr/bin/env bash
#SBATCH --job-name=pz3_dirfield_cache
#SBATCH --array=0-1%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --mem=180G
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%A_%a.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%A_%a.err

set -euo pipefail

REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
SCRIPT_DIR=${REPO}/notebooks/xDESI/abacus_paste
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-/mnt/home/spandey/ceph/env/godmax-jaxhealpy}
CONDA_ENV_HOOK=${CONDA_ENV_HOOK:-${CONDA_ENV_PREFIX}/etc/conda/activate.d/godmax-jaxhealpy.sh}
if [[ -f "${CONDA_ENV_HOOK}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_ENV_HOOK}"
fi
PYTHON=${PYTHON:-${CONDA_ENV_PREFIX}/bin/python}
DRIVER=${SCRIPT_DIR}/stage31_pz1_backlight_validation.py
CONFIG=${CONFIG:-${SCRIPT_DIR}/stage31_pz3_cap600.selected.yaml}

NSIDE=${NSIDE:-1024}
SHELL_INDEX_MOD=${SHELL_INDEX_MOD:-2}
SHELL_INDEX_REM=${SHELL_INDEX_REM:-${SLURM_ARRAY_TASK_ID:-0}}
BATCH_PARENT_PIXELS=${BATCH_PARENT_PIXELS:-262144}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib_${USER}_${SLURM_JOB_ID}_${SHELL_INDEX_REM}}

mkdir -p "${MPLCONFIGDIR}" "${SCRIPT_DIR}/slurm_logs"
cd "${REPO}"

echo "[direct-field-cache] host=$(hostname)"
echo "[direct-field-cache] config=${CONFIG}"
echo "[direct-field-cache] nside=${NSIDE}"
echo "[direct-field-cache] task=${SHELL_INDEX_REM} mod=${SHELL_INDEX_MOD}"
echo "[direct-field-cache] batch_parent_pixels=${BATCH_PARENT_PIXELS}"

srun --ntasks=1 --nodes=1 "${PYTHON}" "${DRIVER}" cache-direct-field-shells \
  --config "${CONFIG}" \
  --nside "${NSIDE}" \
  --shell-index-mod "${SHELL_INDEX_MOD}" \
  --shell-index-rem "${SHELL_INDEX_REM}" \
  --batch-parent-pixels "${BATCH_PARENT_PIXELS}"
