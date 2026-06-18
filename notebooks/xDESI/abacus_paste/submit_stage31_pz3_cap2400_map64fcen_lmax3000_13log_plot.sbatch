#!/usr/bin/env bash
#SBATCH --job-name=map64fcen_plot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.err

# Measure the combined 64-param-fcen MAP-bestfit (linear-N_cen) pasted maps at the
# 13-log / lmax3000 binning, build ONLY the simple 1h+2h (power-add) sim-matched
# theory, and produce the data + theory + sim D_ell comparison plot with the
# 64-param likelihood scale cuts grayed and a log multipole axis. No response curve.
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

CONFIG=${CONFIG:-${SCRIPT_DIR}/stage31_pz3_cap2400_map64fcen_mmin11p147538_lmax3000_13log.selected.yaml}
RUN_ROOT=${RUN_ROOT:-${REPO}/data/xDESI/processed/abacus_backlight/stage31_pz3_cap2400_map64fcen_lmax3000_13log}
NSIDE=${NSIDE:-2048}
LMAX=${LMAX:-3000}
PLOT_ELL_MAX=${PLOT_ELL_MAX:-3000}
PLOT_XSCALE=${PLOT_XSCALE:-log}
KSZ_YLIM_MIN=${KSZ_YLIM_MIN:--5e-5}
KSZ_YLIM_MAX=${KSZ_YLIM_MAX:-5e-5}

SIM_MEAS=${SIM_MEAS:-${RUN_ROOT}/measurements/sim_pz3_cap2400_map64fcen_nside${NSIDE}_lmax${LMAX}_nbin13_log.h5}
SUM_THEORY=${SUM_THEORY:-${RUN_ROOT}/theory/stage31_pz3_cap2400_map64fcen_lmax3000_13log_theory_poweradd_sum_for_sim_measurement_matched_transfers.h5}
VARIANT_PLOT=${VARIANT_PLOT:-${RUN_ROOT}/plots/stage31_pz3_cap2400_map64fcen_lmax3000_13log_full_data_theory_1h2h_with_cap_sim_Dell.pdf}

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

mkdir -p "${MPLCONFIGDIR}" "${SCRIPT_DIR}/slurm_logs" \
  "${RUN_ROOT}/measurements" "${RUN_ROOT}/theory" "${RUN_ROOT}/plots"
cd "${REPO}"

echo "[map64fcen-plot] host=$(hostname)"
echo "[map64fcen-plot] config=${CONFIG}"
echo "[map64fcen-plot] sim_meas=${SIM_MEAS}"
echo "[map64fcen-plot] variant_plot=${VARIANT_PLOT}"

# 1. Measure the combined pasted maps at the 13-log / lmax3000 binning.
srun --ntasks=1 --nodes=1 --cpus-per-task=8 --cpu-bind=none "${PYTHON}" "${DRIVER}" measure-sim \
  --config "${CONFIG}" \
  --nside "${NSIDE}" \
  --output "${SIM_MEAS}" \
  --overwrite

# 2. Simple 1h+2h (power-add) sim-matched theory ONLY (no response transition model).
srun --ntasks=1 --nodes=1 --cpus-per-task=8 --cpu-bind=none "${PYTHON}" "${DRIVER}" theory \
  --config "${CONFIG}" \
  --nside "${NSIDE}" \
  --measurement "${SIM_MEAS}" \
  --transition-model poweradd \
  --sim-matched-transfers \
  --output "${SUM_THEORY}"

# 3. Data + 1h+2h theory + sim in one D_ell plot (log multipole axis; 64-param
#    likelihood bands grayed). --response-theory omitted -> only the 1h+2h curve.
srun --ntasks=1 --nodes=1 --cpus-per-task=8 --cpu-bind=none "${PYTHON}" "${DRIVER}" plot-full-data-theory-variants \
  --config "${CONFIG}" \
  --nside "${NSIDE}" \
  --sum-theory "${SUM_THEORY}" \
  --sim "${SIM_MEAS}" \
  --output "${VARIANT_PLOT}" \
  --plot-ell-max "${PLOT_ELL_MAX}" \
  --gray-unused-bands \
  --plot-xscale "${PLOT_XSCALE}" \
  --ksz-ylim="${KSZ_YLIM_MIN}" "${KSZ_YLIM_MAX}"

echo "[map64fcen-plot] done -> ${VARIANT_PLOT}"
