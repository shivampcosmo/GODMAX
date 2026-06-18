#!/usr/bin/env bash
#SBATCH --job-name=pz3cap2400_13log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.err

# Re-measure the EXISTING checkpoint_000550 pasted maps at the MAP run's 13-log /
# lmax3000 binning, rebuild sim-matched theory, and plot with the survey
# likelihood scale cuts grayed out (matches map_bestfit_full_dell_comparison).
# No re-paste: the painted maps are binning-independent and reused via --maps.

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

CONFIG=${CONFIG:-${SCRIPT_DIR}/stage31_pz3_cap2400_hmcbestfit_mmin11p147538_lmax3000_13log.selected.yaml}
NSIDE=${NSIDE:-2048}
PLOT_ELL_MAX=${PLOT_ELL_MAX:-3000}
KSZ_YLIM_MIN=${KSZ_YLIM_MIN:--5e-5}
KSZ_YLIM_MAX=${KSZ_YLIM_MAX:-5e-5}

# Existing combined pasted map from the checkpoint_000550 paste run (reused, not re-pasted).
COMBINED_MAP=${COMBINED_MAP:-${REPO}/data/xDESI/processed/abacus_backlight/stage31_pz3_cap2400_lmax3000_gk1000_60param_warm100_checkpoint_000550/maps/stage31_pz3_cap2400_hmcbestfit_mmin11p147538_nside2048_lmax4096/abacus_pasted_maps_pz3cap2400_hmcbestfit_z0p63_0p98_logMgt11p147538_nside2048.h5}

OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO}/data/xDESI/processed/abacus_backlight/stage31_pz3_cap2400_hmcbestfit_mmin11p147538_lmax3000_13log}
SIM_MEAS=${SIM_MEAS:-${OUTPUT_ROOT}/measurements/sim_pz3_cap2400_hmcbestfit_mmin11p147538_nside${NSIDE}_lmax3000_nbin13_log.h5}
SUM_THEORY=${SUM_THEORY:-${OUTPUT_ROOT}/theory/stage31_pz3_cap2400_hmcbestfit_mmin11p147538_lmax3000_13log_theory_poweradd_sum_for_sim_measurement_matched_transfers.h5}
RESPONSE_THEORY=${RESPONSE_THEORY:-${OUTPUT_ROOT}/theory/stage31_pz3_cap2400_hmcbestfit_mmin11p147538_lmax3000_13log_theory_response_for_sim_measurement_matched_transfers.h5}
VARIANT_PLOT=${VARIANT_PLOT:-${OUTPUT_ROOT}/plots/stage31_pz3_cap2400_hmcbestfit_mmin11p147538_lmax3000_13log_full_data_theory_variants_with_cap_sim_Dell.pdf}

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
  "${OUTPUT_ROOT}/measurements" "${OUTPUT_ROOT}/theory" "${OUTPUT_ROOT}/plots"
cd "${REPO}"

echo "[13log] host=$(hostname)"
echo "[13log] config=${CONFIG}"
echo "[13log] combined_map=${COMBINED_MAP}"
echo "[13log] sim_meas=${SIM_MEAS}"

# 1. Re-measure the existing pasted maps at the 13-log / lmax3000 binning.
srun --ntasks=1 --nodes=1 --cpus-per-task=8 --cpu-bind=none "${PYTHON}" "${DRIVER}" measure-sim \
  --config "${CONFIG}" \
  --nside "${NSIDE}" \
  --maps "${COMBINED_MAP}" \
  --output "${SIM_MEAS}" \
  --overwrite

# 2. Sim-matched theory variants at the same 13-log binning (reads the 13-band measurement).
srun --ntasks=1 --nodes=1 --cpus-per-task=8 --cpu-bind=none "${PYTHON}" "${DRIVER}" theory \
  --config "${CONFIG}" \
  --nside "${NSIDE}" \
  --measurement "${SIM_MEAS}" \
  --transition-model poweradd \
  --sim-matched-transfers \
  --output "${SUM_THEORY}"

srun --ntasks=1 --nodes=1 --cpus-per-task=8 --cpu-bind=none "${PYTHON}" "${DRIVER}" theory \
  --config "${CONFIG}" \
  --nside "${NSIDE}" \
  --measurement "${SIM_MEAS}" \
  --transition-model response \
  --sim-matched-transfers \
  --output "${RESPONSE_THEORY}"

# 3. Plot: 13-band data points (from config godmax.measurement_h5) + sim + theory variants,
#    with bands outside the likelihood scale cuts grayed out (config likelihood_cuts).
srun --ntasks=1 --nodes=1 --cpus-per-task=8 --cpu-bind=none "${PYTHON}" "${DRIVER}" plot-full-data-theory-variants \
  --config "${CONFIG}" \
  --nside "${NSIDE}" \
  --sum-theory "${SUM_THEORY}" \
  --response-theory "${RESPONSE_THEORY}" \
  --sim "${SIM_MEAS}" \
  --output "${VARIANT_PLOT}" \
  --plot-ell-max "${PLOT_ELL_MAX}" \
  --gray-unused-bands \
  --plot-xscale "${PLOT_XSCALE:-log}" \
  --ksz-ylim="${KSZ_YLIM_MIN}" "${KSZ_YLIM_MAX}"

echo "[13log] done -> ${VARIANT_PLOT}"
