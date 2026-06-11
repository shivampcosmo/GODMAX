#!/usr/bin/env bash
#SBATCH --job-name=pz3_plus_direct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=160G
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

CONFIG=${CONFIG:?CONFIG must point to the Stage31 cap config}
MAPS=${MAPS:?MAPS must point to the pasted-plus-direct-field HDF5}
SIM_MEAS=${SIM_MEAS:?SIM_MEAS output path is required}
SUM_THEORY=${SUM_THEORY:?SUM_THEORY output path is required}
RESPONSE_THEORY=${RESPONSE_THEORY:?RESPONSE_THEORY output path is required}
VARIANT_PLOT=${VARIANT_PLOT:?VARIANT_PLOT output path is required}
NSIDE=${NSIDE:-2048}
SIM_MATCHED_TRANSFERS=${SIM_MATCHED_TRANSFERS:-1}
KSZ_YLIM_MIN=${KSZ_YLIM_MIN:--5e-5}
KSZ_YLIM_MAX=${KSZ_YLIM_MAX:-5e-5}
THEORY_TRANSFER_ARGS=()
if [ "${SIM_MATCHED_TRANSFERS}" -gt 0 ]; then
  THEORY_TRANSFER_ARGS+=(--sim-matched-transfers)
fi

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib_${USER}_${SLURM_JOB_ID}}
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
export PASTE_JAX_PLATFORMS=${PASTE_JAX_PLATFORMS:-cpu}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p "${MPLCONFIGDIR}" "${SCRIPT_DIR}/slurm_logs" "$(dirname "${SIM_MEAS}")" "$(dirname "${SUM_THEORY}")" "$(dirname "${VARIANT_PLOT}")"
cd "${REPO}"

echo "[plus-direct] host=$(hostname)"
echo "[plus-direct] config=${CONFIG}"
echo "[plus-direct] maps=${MAPS}"
echo "[plus-direct] sim_meas=${SIM_MEAS}"
echo "[plus-direct] sum_theory=${SUM_THEORY}"
echo "[plus-direct] response_theory=${RESPONSE_THEORY}"
echo "[plus-direct] variant_plot=${VARIANT_PLOT}"
echo "[plus-direct] sim_matched_transfers=${SIM_MATCHED_TRANSFERS}"
echo "[plus-direct] ksz_ylim=${KSZ_YLIM_MIN} ${KSZ_YLIM_MAX}"

srun --ntasks=1 --nodes=1 --cpus-per-task=8 --cpu-bind=none "${PYTHON}" "${DRIVER}" measure-sim \
  --config "${CONFIG}" \
  --nside "${NSIDE}" \
  --maps "${MAPS}" \
  --output "${SIM_MEAS}" \
  --overwrite

srun --ntasks=1 --nodes=1 --cpus-per-task=8 --cpu-bind=none "${PYTHON}" "${DRIVER}" theory \
  --config "${CONFIG}" \
  --nside "${NSIDE}" \
  --measurement "${SIM_MEAS}" \
  --transition-model poweradd \
  "${THEORY_TRANSFER_ARGS[@]}" \
  --output "${SUM_THEORY}"

srun --ntasks=1 --nodes=1 --cpus-per-task=8 --cpu-bind=none "${PYTHON}" "${DRIVER}" theory \
  --config "${CONFIG}" \
  --nside "${NSIDE}" \
  --measurement "${SIM_MEAS}" \
  --transition-model response \
  "${THEORY_TRANSFER_ARGS[@]}" \
  --output "${RESPONSE_THEORY}"

srun --ntasks=1 --nodes=1 --cpus-per-task=8 --cpu-bind=none "${PYTHON}" "${DRIVER}" plot-full-data-theory-variants \
  --config "${CONFIG}" \
  --nside "${NSIDE}" \
  --sum-theory "${SUM_THEORY}" \
  --response-theory "${RESPONSE_THEORY}" \
  --sim "${SIM_MEAS}" \
  --output "${VARIANT_PLOT}" \
  --ksz-ylim "${KSZ_YLIM_MIN}" "${KSZ_YLIM_MAX}"

echo "[plus-direct] done"
