#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=xdesi_hmc_ckpt_monitor
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/ceph/users/spandey/ltu-godmax/GODMAX}"
PYTHON="${PYTHON:-/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python}"
CHECKPOINT_MONITOR="${CHECKPOINT_MONITOR:-${REPO_ROOT}/notebooks/xDESI/survey_measure/monitor_godmax_hmc_stage31_checkpoints.py}"
COMBINER="${COMBINER:-${REPO_ROOT}/notebooks/xDESI/survey_measure/combine_godmax_hmc_stage31_workers.py}"
CONFIG="${CONFIG:?CONFIG is required}"
WORKER_DIR="${WORKER_DIR:?WORKER_DIR is required}"
COMBINED_DIR="${COMBINED_DIR:?COMBINED_DIR is required}"
COMBINED_SUFFIX="${COMBINED_SUFFIX:?COMBINED_SUFFIX is required}"
RUN_LABEL="${RUN_LABEL:?RUN_LABEL is required}"
MONITOR_STOP_FILE="${MONITOR_STOP_FILE:?MONITOR_STOP_FILE is required}"
WATCH_JOB_ID="${WATCH_JOB_ID:-}"
EXPECTED_WORKERS="${EXPECTED_WORKERS:-4}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"

CHECKPOINT_PASTE_AFTER="${CHECKPOINT_PASTE_AFTER:-1}"
CHECKPOINT_PZ3_GATE="${CHECKPOINT_PZ3_GATE:-${REPO_ROOT}/notebooks/xDESI/abacus_paste/submit_stage31_pz3_cap2400_hmcbestfit_gate.sbatch}"
CHECKPOINT_PASTE_CONFIG_TEMPLATE="${CHECKPOINT_PASTE_CONFIG_TEMPLATE:-${REPO_ROOT}/notebooks/xDESI/abacus_paste/stage31_pz3_cap2400_hmcbestfit_mmin11p147538_nside2048_lmax4096.selected.yaml}"
CHECKPOINT_PASTE_RUN_ROOT_BASE="${CHECKPOINT_PASTE_RUN_ROOT_BASE:-${REPO_ROOT}/data/xDESI/processed/abacus_backlight/stage31_pz3_cap2400_hmccheckpoints_mmin11p147538}"
CHECKPOINT_PASTE_NSIDE="${CHECKPOINT_PASTE_NSIDE:-2048}"
CHECKPOINT_PASTE_LMAX="${CHECKPOINT_PASTE_LMAX:-4096}"
CHECKPOINT_PASTE_NUM_SPLITS="${CHECKPOINT_PASTE_NUM_SPLITS:-4}"
CHECKPOINT_PASTE_PIXEL_WORKERS="${CHECKPOINT_PASTE_PIXEL_WORKERS:-16}"
CHECKPOINT_PASTE_CATALOG_SOURCE="${CHECKPOINT_PASTE_CATALOG_SOURCE:-${REPO_ROOT}/data/xDESI/processed/abacus_backlight/stage31_pz3_cap2400_hmcfailed_mmin11p147538/halos/abacus_c9999_ph9999_pz3cap2400_hmcfailed_z0p63_0p98_logMgt11p147538_halos.h5}"
CHECKPOINT_PASTE_CATALOG_OUTPUT_NAME="${CHECKPOINT_PASTE_CATALOG_OUTPUT_NAME:-abacus_c9999_ph9999_pz3cap2400_hmcbestfit_z0p63_0p98_logMgt11p147538_halos.h5}"
CHECKPOINT_PASTE_DO_PREPROCESS="${CHECKPOINT_PASTE_DO_PREPROCESS:-0}"
CHECKPOINT_PASTE_DO_PASTED_THEORY="${CHECKPOINT_PASTE_DO_PASTED_THEORY:-1}"
CHECKPOINT_PASTE_DO_DIRECT_FIELD="${CHECKPOINT_PASTE_DO_DIRECT_FIELD:-0}"
CHECKPOINT_PASTE_DO_PLUS_DIRECT="${CHECKPOINT_PASTE_DO_PLUS_DIRECT:-0}"
CHECKPOINT_PASTE_SIM_MATCHED_TRANSFERS="${CHECKPOINT_PASTE_SIM_MATCHED_TRANSFERS:-1}"
KSZ_VELOCITY_MODE="${KSZ_VELOCITY_MODE:-photoz_reconstruction_emulation}"
KSZ_RECONSTRUCTION_NOISE_SEED="${KSZ_RECONSTRUCTION_NOISE_SEED:-12345}"
KSZ_YLIM_MIN="${KSZ_YLIM_MIN:--5e-5}"
KSZ_YLIM_MAX="${KSZ_YLIM_MAX:-5e-5}"
PLOT_ELL_MAX="${PLOT_ELL_MAX:-2800}"

cd "${REPO_ROOT}"
mkdir -p "$(dirname "${MONITOR_STOP_FILE}")"
rm -f "${MONITOR_STOP_FILE}"

export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

args=(
  "${CHECKPOINT_MONITOR}"
  --config "${CONFIG}"
  --worker-dir "${WORKER_DIR}"
  --combined-dir "${COMBINED_DIR}"
  --combined-suffix "${COMBINED_SUFFIX}"
  --run-label "${RUN_LABEL}"
  --expected-workers "${EXPECTED_WORKERS}"
  --poll-interval "${POLL_INTERVAL}"
  --stop-file "${MONITOR_STOP_FILE}"
  --combiner "${COMBINER}"
  --python "${PYTHON}"
  --paste-gate "${CHECKPOINT_PZ3_GATE}"
  --paste-config-template "${CHECKPOINT_PASTE_CONFIG_TEMPLATE}"
  --paste-run-root-base "${CHECKPOINT_PASTE_RUN_ROOT_BASE}"
  --nside "${CHECKPOINT_PASTE_NSIDE}"
  --lmax "${CHECKPOINT_PASTE_LMAX}"
  --num-splits "${CHECKPOINT_PASTE_NUM_SPLITS}"
  --pixel-workers "${CHECKPOINT_PASTE_PIXEL_WORKERS}"
  --ksz-ylim-min="${KSZ_YLIM_MIN}"
  --ksz-ylim-max="${KSZ_YLIM_MAX}"
  --plot-ell-max "${PLOT_ELL_MAX}"
  --ksz-velocity-mode "${KSZ_VELOCITY_MODE}"
  --ksz-reconstruction-noise-seed "${KSZ_RECONSTRUCTION_NOISE_SEED}"
  --sim-matched-transfers "${CHECKPOINT_PASTE_SIM_MATCHED_TRANSFERS}"
  --do-preprocess "${CHECKPOINT_PASTE_DO_PREPROCESS}"
  --do-pasted-theory "${CHECKPOINT_PASTE_DO_PASTED_THEORY}"
  --do-direct-field "${CHECKPOINT_PASTE_DO_DIRECT_FIELD}"
  --do-plus-direct "${CHECKPOINT_PASTE_DO_PLUS_DIRECT}"
  --catalog-source "${CHECKPOINT_PASTE_CATALOG_SOURCE}"
  --catalog-output-name "${CHECKPOINT_PASTE_CATALOG_OUTPUT_NAME}"
  --postprocess-platform cpu
  --retry-failed
)
if [[ "${CHECKPOINT_PASTE_AFTER}" == "1" ]]; then
  args+=(--submit-paste)
fi

echo "[$(date)] starting CPU checkpoint monitor"
"${PYTHON}" -u "${args[@]}" &
monitor_pid="$!"

if [[ -n "${WATCH_JOB_ID}" ]]; then
  while squeue -h -j "${WATCH_JOB_ID}" | grep -q .; do
    sleep "${POLL_INTERVAL}"
  done
  echo "[$(date)] watched job ${WATCH_JOB_ID} left queue; stopping monitor"
  touch "${MONITOR_STOP_FILE}"
fi

wait "${monitor_pid}"
echo "[$(date)] CPU checkpoint monitor done"
