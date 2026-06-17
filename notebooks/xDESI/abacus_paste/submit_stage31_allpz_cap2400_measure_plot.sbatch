#!/usr/bin/env bash
#SBATCH --job-name=allpz_cap2400_meas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --mem=192G
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
SCRIPT_DIR=${REPO}/notebooks/xDESI/abacus_paste
RUN_ROOT=${RUN_ROOT:-${REPO}/data/xDESI/processed/abacus_backlight/stage31_allpz_cap2400_hmcfailed_mmin11p147538}
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-/mnt/home/spandey/ceph/env/godmax-jaxhealpy}
CONDA_ENV_HOOK=${CONDA_ENV_HOOK:-${CONDA_ENV_PREFIX}/etc/conda/activate.d/godmax-jaxhealpy.sh}
if [[ -f "${CONDA_ENV_HOOK}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_ENV_HOOK}"
fi
PYTHON=${PYTHON:-${CONDA_ENV_PREFIX}/bin/python}
MEASURE_DRIVER=${SCRIPT_DIR}/stage31_allpz_cap_measure.py
PLOT_DRIVER=${SCRIPT_DIR}/stage31_pz1_backlight_validation.py

CONFIG=${CONFIG:?CONFIG is required}
PZ1_CONFIG=${PZ1_CONFIG:?PZ1_CONFIG is required}
PZ2_CONFIG=${PZ2_CONFIG:?PZ2_CONFIG is required}
PZ3_CONFIG=${PZ3_CONFIG:?PZ3_CONFIG is required}
PZ4_CONFIG=${PZ4_CONFIG:?PZ4_CONFIG is required}
CONTINUOUS_MAPS=${CONTINUOUS_MAPS:?CONTINUOUS_MAPS is required}
PZ1_MAPS=${PZ1_MAPS:?PZ1_MAPS is required}
PZ2_MAPS=${PZ2_MAPS:?PZ2_MAPS is required}
PZ3_MAPS=${PZ3_MAPS:?PZ3_MAPS is required}
PZ4_MAPS=${PZ4_MAPS:?PZ4_MAPS is required}

NSIDE=${NSIDE:-2048}
LMAX=${LMAX:-4096}
NBIN=${NBIN:-10}
BINNING=${BINNING:-linear}
KSZ_YLIM_MIN=${KSZ_YLIM_MIN:--5e-5}
KSZ_YLIM_MAX=${KSZ_YLIM_MAX:-5e-5}
DO_PLOTS=${DO_PLOTS:-1}

HMC_COMBINED=${HMC_COMBINED:-${REPO}/notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_midres2048_true_nz_hmc_stage31_multigpu/stage31_hmc_abacus_cosmo_midres2048_simple1h2h_lmax4096_gk1024_mmin11p147538_1600x16_v1/combined}
HMC_SUFFIX=${HMC_SUFFIX:-stage31_multigpu_abacus_cosmo_midres2048_simple1h2h_lmax4096_gk1024_mmin11p147538_1600x16_v1}
BESTFIT_VECTOR=${BESTFIT_VECTOR:-${HMC_COMBINED}/bestfit_full_theory_data_vector_${HMC_SUFFIX}.npz}
FIDUCIAL_VECTOR=${FIDUCIAL_VECTOR:-${BESTFIT_VECTOR}}

MEAS_DIR=${MEAS_DIR:-${RUN_ROOT}/measurements}
PLOT_DIR=${PLOT_DIR:-${RUN_ROOT}/plots}
ALLPZ_MEAS=${ALLPZ_MEAS:-${MEAS_DIR}/sim_allpz_cap2400_hmcfailed_allz_continuous_mmin11p147538_vrec_emulated_nside${NSIDE}_lmax${LMAX}_nbin${NBIN}_${BINNING}.h5}
PZ1_MEAS=${PZ1_MEAS:-${MEAS_DIR}/sim_pz1_cap2400_hmcfailed_allz_continuous_mmin11p147538_vrec_emulated_nside${NSIDE}_lmax${LMAX}_nbin${NBIN}_${BINNING}.h5}
PZ2_MEAS=${PZ2_MEAS:-${MEAS_DIR}/sim_pz2_cap2400_hmcfailed_allz_continuous_mmin11p147538_vrec_emulated_nside${NSIDE}_lmax${LMAX}_nbin${NBIN}_${BINNING}.h5}
PZ3_MEAS=${PZ3_MEAS:-${MEAS_DIR}/sim_pz3_cap2400_hmcfailed_allz_continuous_mmin11p147538_vrec_emulated_nside${NSIDE}_lmax${LMAX}_nbin${NBIN}_${BINNING}.h5}
PZ4_MEAS=${PZ4_MEAS:-${MEAS_DIR}/sim_pz4_cap2400_hmcfailed_allz_continuous_mmin11p147538_vrec_emulated_nside${NSIDE}_lmax${LMAX}_nbin${NBIN}_${BINNING}.h5}

PZ1_PLOT=${PZ1_PLOT:-${PLOT_DIR}/stage31_pz1_cap2400_hmcfailed_allz_continuous_mmin11p147538_full_area_data_bestfit_with_cap_sim_vrec_emulated_nside${NSIDE}_lmax${LMAX}_Dell.pdf}
PZ2_PLOT=${PZ2_PLOT:-${PLOT_DIR}/stage31_pz2_cap2400_hmcfailed_allz_continuous_mmin11p147538_full_area_data_bestfit_with_cap_sim_vrec_emulated_nside${NSIDE}_lmax${LMAX}_Dell.pdf}
PZ3_PLOT=${PZ3_PLOT:-${PLOT_DIR}/stage31_pz3_cap2400_hmcfailed_allz_continuous_mmin11p147538_full_area_data_bestfit_with_cap_sim_vrec_emulated_nside${NSIDE}_lmax${LMAX}_Dell.pdf}
PZ4_PLOT=${PZ4_PLOT:-${PLOT_DIR}/stage31_pz4_cap2400_hmcfailed_allz_continuous_mmin11p147538_full_area_data_bestfit_with_cap_sim_vrec_emulated_nside${NSIDE}_lmax${LMAX}_Dell.pdf}
COMBINED_PLOT=${COMBINED_PLOT:-${PLOT_DIR}/stage31_allpz_cap2400_hmcfailed_allz_continuous_mmin11p147538_full_area_data_bestfit_with_cap_sim_vrec_emulated_nside${NSIDE}_lmax${LMAX}_Dell.pdf}

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib_${USER}_${SLURM_JOB_ID}}
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
export PASTE_JAX_PLATFORMS=${PASTE_JAX_PLATFORMS:-cpu}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p "${MPLCONFIGDIR}" "${SCRIPT_DIR}/slurm_logs" "${MEAS_DIR}" "${PLOT_DIR}"
cd "${REPO}"

echo "[allpz-measure] host=$(hostname)"
echo "[allpz-measure] config=${CONFIG}"
echo "[allpz-measure] continuous_maps=${CONTINUOUS_MAPS}"
echo "[allpz-measure] pz maps=${PZ1_MAPS} ${PZ2_MAPS} ${PZ3_MAPS} ${PZ4_MAPS}"
echo "[allpz-measure] allpz_meas=${ALLPZ_MEAS}"
echo "[allpz-measure] bestfit_vector=${BESTFIT_VECTOR}"
echo "[allpz-measure] ksz_ylim=${KSZ_YLIM_MIN} ${KSZ_YLIM_MAX}"

srun --ntasks=1 --nodes=1 --cpus-per-task=12 --cpu-bind=none \
  "${PYTHON}" "${MEASURE_DRIVER}" \
  --config "${CONFIG}" \
  --continuous-maps "${CONTINUOUS_MAPS}" \
  --galaxy-map "pz1=${PZ1_MAPS}" \
  --galaxy-map "pz2=${PZ2_MAPS}" \
  --galaxy-map "pz3=${PZ3_MAPS}" \
  --galaxy-map "pz4=${PZ4_MAPS}" \
  --output "${ALLPZ_MEAS}" \
  --per-pz-output "pz1=${PZ1_MEAS}" \
  --per-pz-output "pz2=${PZ2_MEAS}" \
  --per-pz-output "pz3=${PZ3_MEAS}" \
  --per-pz-output "pz4=${PZ4_MEAS}" \
  --nside "${NSIDE}" \
  --overwrite

if [ "${DO_PLOTS}" -gt 0 ]; then
  for pz in 1 2 3 4; do
    cfg_var="PZ${pz}_CONFIG"
    meas_var="PZ${pz}_MEAS"
    plot_var="PZ${pz}_PLOT"
    cfg="${!cfg_var}"
    meas="${!meas_var}"
    plot="${!plot_var}"
    echo "[allpz-measure] plotting pz${pz}: ${plot}"
    srun --ntasks=1 --nodes=1 --cpus-per-task=4 --cpu-bind=none \
      "${PYTHON}" "${PLOT_DRIVER}" plot-full-data \
      --config "${cfg}" \
      --sim "${meas}" \
      --fiducial-vector "${FIDUCIAL_VECTOR}" \
      --bestfit-vector "${BESTFIT_VECTOR}" \
      --output "${plot}" \
      --dell \
      --ksz-ylim="${KSZ_YLIM_MIN}" "${KSZ_YLIM_MAX}" \
      --no-include-fiducial
  done
  if command -v pdfunite >/dev/null 2>&1; then
    pdfunite "${PZ1_PLOT}" "${PZ2_PLOT}" "${PZ3_PLOT}" "${PZ4_PLOT}" "${COMBINED_PLOT}"
    echo "[allpz-measure] combined_plot=${COMBINED_PLOT}"
  else
    echo "[allpz-measure] pdfunite not available; per-pz plots were written."
  fi
fi

echo "[allpz-measure] done"
