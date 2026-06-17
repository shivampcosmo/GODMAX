#!/usr/bin/env bash
#SBATCH --job-name=allpz_cap2400_combine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
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
CONFIG=${CONFIG:?CONFIG is required}
NSIDE=${NSIDE:-2048}
NUM_SPLITS=${NUM_SPLITS:-4}
DO_PLOT_MAPS=${DO_PLOT_MAPS:-0}

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib_${USER}_${SLURM_JOB_ID}}
mkdir -p "${MPLCONFIGDIR}" "${SCRIPT_DIR}/slurm_logs"
cd "${REPO}"

echo "[allpz-combine] host=$(hostname)"
echo "[allpz-combine] config=${CONFIG}"
echo "[allpz-combine] python=${PYTHON}"
echo "[allpz-combine] nside=${NSIDE} num_splits=${NUM_SPLITS}"
echo "[allpz-combine] do_plot_maps=${DO_PLOT_MAPS}"

srun --ntasks=1 --nodes=1 --cpus-per-task=8 --cpu-bind=none \
  "${PYTHON}" "${DRIVER}" combine-maps \
  --config "${CONFIG}" \
  --nside "${NSIDE}" \
  --num-splits "${NUM_SPLITS}" \
  --overwrite

if [ "${DO_PLOT_MAPS}" -gt 0 ]; then
  srun --ntasks=1 --nodes=1 --cpus-per-task=8 --cpu-bind=none \
    "${PYTHON}" "${DRIVER}" plot-healpix-maps \
    --config "${CONFIG}" \
    --nside "${NSIDE}"
fi

echo "[allpz-combine] done"
