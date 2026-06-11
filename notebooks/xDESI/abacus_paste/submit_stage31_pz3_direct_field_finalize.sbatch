#!/usr/bin/env bash
#SBATCH --job-name=pz3_dirfield_done
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.err

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
NUM_SPLITS=${NUM_SPLITS:-4}
BATCH_PARENT_PIXELS=${BATCH_PARENT_PIXELS:-262144}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib_${USER}_${SLURM_JOB_ID}}

mkdir -p "${MPLCONFIGDIR}" "${SCRIPT_DIR}/slurm_logs"
cd "${REPO}"

echo "[direct-field-finalize] host=$(hostname)"
echo "[direct-field-finalize] config=${CONFIG}"
echo "[direct-field-finalize] nside=${NSIDE} num_splits=${NUM_SPLITS}"
echo "[direct-field-finalize] batch_parent_pixels=${BATCH_PARENT_PIXELS}"

srun --ntasks=1 --nodes=1 "${PYTHON}" "${DRIVER}" combine-maps \
  --config "${CONFIG}" \
  --nside "${NSIDE}" \
  --num-splits "${NUM_SPLITS}" \
  --overwrite

srun --ntasks=1 --nodes=1 "${PYTHON}" "${DRIVER}" build-direct-field-map \
  --config "${CONFIG}" \
  --nside "${NSIDE}" \
  --batch-parent-pixels "${BATCH_PARENT_PIXELS}" \
  --overwrite
