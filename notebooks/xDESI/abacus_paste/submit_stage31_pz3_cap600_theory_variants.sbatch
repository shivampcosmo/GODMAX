#!/usr/bin/env bash
#SBATCH --job-name=pz3cap600_theory
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
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
CONFIG=${CONFIG:-${SCRIPT_DIR}/stage31_pz3_cap600_mmin11p147538_nside2048_lmax4096.selected.yaml}

RUN_ROOT=${RUN_ROOT:-${REPO}/data/xDESI/processed/abacus_backlight/stage31_pz3_cap600_mmin11p147538}
NSIDE=${NSIDE:-2048}
LMAX=${LMAX:-4096}
SIM_MEAS=${SIM_MEAS:-${RUN_ROOT}/measurements/sim_pz3_cap600_mmin11p147538_nside${NSIDE}_lmax${LMAX}_nbin10_linear.h5}
SUM_THEORY=${SUM_THEORY:-${RUN_ROOT}/theory/stage31_pz3_cap600_mmin11p147538_nside${NSIDE}_lmax${LMAX}_theory_poweradd_sum_for_sim_measurement.h5}
RESPONSE_THEORY=${RESPONSE_THEORY:-${RUN_ROOT}/theory/stage31_pz3_cap600_mmin11p147538_nside${NSIDE}_lmax${LMAX}_theory_response_for_sim_measurement.h5}
VARIANT_PLOT=${VARIANT_PLOT:-${RUN_ROOT}/plots/stage31_pz3_cap600_mmin11p147538_nside${NSIDE}_lmax${LMAX}_full_data_theory_variants_with_cap_sim_Dell.pdf}
SIM_MATCHED_TRANSFERS=${SIM_MATCHED_TRANSFERS:-1}
KSZ_YLIM_MIN=${KSZ_YLIM_MIN:--5e-5}
KSZ_YLIM_MAX=${KSZ_YLIM_MAX:-5e-5}
PLOT_ELL_MAX=${PLOT_ELL_MAX:-2800}
THEORY_TRANSFER_ARGS=()
if [ "${SIM_MATCHED_TRANSFERS}" -gt 0 ]; then
  THEORY_TRANSFER_ARGS+=(--sim-matched-transfers)
fi

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib_${USER}_${SLURM_JOB_ID}}
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
export PASTE_JAX_PLATFORMS=${PASTE_JAX_PLATFORMS:-cpu}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
unset SLURM_JOB_GRES SLURM_STEP_GRES SLURM_GPUS SLURM_STEP_GPUS SLURM_JOB_GPUS SLURM_STEP_GPUS CUDA_VISIBLE_DEVICES

mkdir -p "${MPLCONFIGDIR}" "${SCRIPT_DIR}/slurm_logs" "${RUN_ROOT}/theory" "${RUN_ROOT}/plots"
cd "${REPO}"

echo "[theory] host=$(hostname)"
echo "[theory] config=${CONFIG}"
echo "[theory] python=${PYTHON}"
echo "[theory] sim_meas=${SIM_MEAS}"
echo "[theory] sum_theory=${SUM_THEORY}"
echo "[theory] response_theory=${RESPONSE_THEORY}"
echo "[theory] variant_plot=${VARIANT_PLOT}"
echo "[theory] sim_matched_transfers=${SIM_MATCHED_TRANSFERS}"
echo "[theory] ksz_ylim=${KSZ_YLIM_MIN} ${KSZ_YLIM_MAX}"
echo "[theory] plot_ell_max=${PLOT_ELL_MAX}"

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
  --plot-ell-max "${PLOT_ELL_MAX}" \
  --ksz-ylim="${KSZ_YLIM_MIN}" "${KSZ_YLIM_MAX}"

echo "[theory] done"
