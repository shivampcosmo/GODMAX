#!/usr/bin/env bash
#SBATCH --job-name=pz3cap600_combine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=96G
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
MONITOR=${SCRIPT_DIR}/monitor_node_usage.sh

NSIDE=${NSIDE:-1024}
NUM_SPLITS=${NUM_SPLITS:-4}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-180}
GPU_SAMPLE_INTERVAL=${GPU_SAMPLE_INTERVAL:-10}
LMAX=${LMAX:-1024}
DO_FULL_DATA_PLOT=${DO_FULL_DATA_PLOT:-1}
MAX_L=$((3 * NSIDE - 1))
if [ "${LMAX}" -gt "${MAX_L}" ]; then
  LMAX="${MAX_L}"
fi

SIM_MEAS=${SIM_MEAS:-${REPO}/data/xDESI/processed/abacus_backlight/stage31_pz3_cap600/measurements/sim_pz3_cap600_nside${NSIDE}_lmax${LMAX}_nbin10_linear.h5}
FULL_AREA_PLOT=${FULL_AREA_PLOT:-${REPO}/data/xDESI/processed/abacus_backlight/stage31_pz3_cap600/plots/stage31_pz3_full_area_data_bestfit_with_cap_sim_nside${NSIDE}_lmax${LMAX}_Dell.pdf}
BESTFIT_VECTOR=${BESTFIT_VECTOR:-${REPO}/notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_fast1024_true_nz_hmc_stage31_multigpu/stage31_hmc_8000x16_v2/combined/bestfit_theory_data_vector_stage31_multigpu_v2.npz}
FIDUCIAL_VECTOR=${FIDUCIAL_VECTOR:-${BESTFIT_VECTOR}}
KSZ_YLIM_MIN=${KSZ_YLIM_MIN:--5e-5}
KSZ_YLIM_MAX=${KSZ_YLIM_MAX:-5e-5}
PLOT_ELL_MAX=${PLOT_ELL_MAX:-2800}

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib_${USER}_${SLURM_JOB_ID}}
unset SLURM_JOB_GRES SLURM_STEP_GRES SLURM_GPUS SLURM_STEP_GPUS SLURM_JOB_GPUS SLURM_STEP_GPUS CUDA_VISIBLE_DEVICES
mkdir -p "${MPLCONFIGDIR}" "${SCRIPT_DIR}/slurm_logs"
cd "${REPO}"

echo "[combine] host=$(hostname)"
echo "[combine] config=${CONFIG}"
echo "[combine] python=${PYTHON}"
echo "[combine] nside=${NSIDE} lmax=${LMAX} num_splits=${NUM_SPLITS}"
echo "[combine] sim_meas=${SIM_MEAS}"
echo "[combine] full_area_plot=${FULL_AREA_PLOT}"
echo "[combine] bestfit_vector=${BESTFIT_VECTOR}"
echo "[combine] fiducial_vector=${FIDUCIAL_VECTOR}"
echo "[combine] ksz_ylim=${KSZ_YLIM_MIN} ${KSZ_YLIM_MAX}"
echo "[combine] plot_ell_max=${PLOT_ELL_MAX}"
echo "[combine] do_full_data_plot=${DO_FULL_DATA_PLOT}"
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

srun --ntasks=1 --nodes=1 "${PYTHON}" "${DRIVER}" measure-sim \
  --config "${CONFIG}" \
  --nside "${NSIDE}" \
  --overwrite

if [ "${DO_FULL_DATA_PLOT}" -gt 0 ]; then
  srun --ntasks=1 --nodes=1 "${PYTHON}" "${DRIVER}" plot-full-data \
    --config "${CONFIG}" \
    --sim "${SIM_MEAS}" \
    --fiducial-vector "${FIDUCIAL_VECTOR}" \
    --bestfit-vector "${BESTFIT_VECTOR}" \
    --output "${FULL_AREA_PLOT}" \
    --dell \
    --plot-ell-max "${PLOT_ELL_MAX}" \
    --ksz-ylim="${KSZ_YLIM_MIN}" "${KSZ_YLIM_MAX}" \
    --no-include-fiducial
else
  echo "[combine] skipping plot-full-data because DO_FULL_DATA_PLOT=${DO_FULL_DATA_PLOT}"
fi

echo "[combine] sim_measurement=${SIM_MEAS}"
echo "[combine] full_area_plot=${FULL_AREA_PLOT}"
