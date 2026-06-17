#!/bin/bash
#SBATCH -p gpu
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=15
#SBATCH --mem=800G
#SBATCH --time=24:00:00
#SBATCH --job-name=xdesi_hmc_stage31_4gpu
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%j.err

set -euo pipefail

REPO_ROOT="/mnt/ceph/users/spandey/ltu-godmax/GODMAX"
PYTHON="/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python"
RUNNER="${REPO_ROOT}/notebooks/xDESI/survey_measure/run_godmax_multiprobe_hmc_stage31.py"
COMBINER="${REPO_ROOT}/notebooks/xDESI/survey_measure/combine_godmax_hmc_stage31_workers.py"
CHECKPOINT_MONITOR="${REPO_ROOT}/notebooks/xDESI/survey_measure/monitor_godmax_hmc_stage31_checkpoints.py"
CONFIG="${CONFIG:-${REPO_ROOT}/param_files/xDESI/params_multiprobe_midres2048_hmc_stage31_abacus_cosmo_simple1h2h_lmax3000_gk1000_depth6_defaultacc_warm25_2000_60param.yaml}"
LOG_DIR="${REPO_ROOT}/notebooks/xDESI/survey_measure/logs"

yaml_value() {
  "${PYTHON}" -c '
import sys, yaml
path, dotted = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as handle:
    value = yaml.safe_load(handle)
for key in dotted.split("."):
    value = value[key]
print(value)
' "$1" "$2"
}

yaml_value_optional() {
  "${PYTHON}" -c '
import sys, yaml
path, dotted = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as handle:
    value = yaml.safe_load(handle)
try:
    for key in dotted.split("."):
        value = value[key]
except (KeyError, TypeError):
    value = None
if value is not None:
    print(value)
' "$1" "$2"
}

RUN_VERSION="${RUN_VERSION:-abacus_cosmo_midres2048_simple1h2h_lmax3000_gk1000_mmin11p147538_60param_13log_apo1degC2_pairmean_v1}"
RUN_LABEL="${1:-stage31_hmc_${RUN_VERSION}}"
COMBINED_SUFFIX="${COMBINED_SUFFIX:-stage31_multigpu_${RUN_VERSION}}"
N_WORKERS="${N_WORKERS:-4}"
CHAINS_PER_GPU="${CHAINS_PER_GPU:-$(yaml_value "${CONFIG}" sampler.num_chains)}"
NUM_WARMUP="${NUM_WARMUP:-$(yaml_value "${CONFIG}" sampler.num_warmup)}"
NUM_SAMPLES="${NUM_SAMPLES:-$(yaml_value "${CONFIG}" sampler.num_samples)}"
MAX_TREE_DEPTH="${MAX_TREE_DEPTH:-$(yaml_value "${CONFIG}" sampler.max_tree_depth)}"
TARGET_ACCEPT="${TARGET_ACCEPT:-$(yaml_value_optional "${CONFIG}" sampler.target_accept_prob)}"
BASE_SEED="${BASE_SEED:-42000}"
GPU_SANITY_CHECK="${GPU_SANITY_CHECK:-1}"
COMBINE_AFTER="${COMBINE_AFTER:-1}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-120}"
CHECKPOINT_SAMPLES_EVERY="${CHECKPOINT_SAMPLES_EVERY:-25}"
CHECKPOINT_COMBINE_AFTER="${CHECKPOINT_COMBINE_AFTER:-1}"
CHECKPOINT_PASTE_AFTER="${CHECKPOINT_PASTE_AFTER:-0}"
CHECKPOINT_MONITOR_INTERVAL="${CHECKPOINT_MONITOR_INTERVAL:-60}"
CHECKPOINT_PZ3_GATE="${CHECKPOINT_PZ3_GATE:-${REPO_ROOT}/notebooks/xDESI/abacus_paste/submit_stage31_pz3_cap2400_hmcbestfit_gate.sbatch}"
CHECKPOINT_PASTE_CONFIG_TEMPLATE="${CHECKPOINT_PASTE_CONFIG_TEMPLATE:-${REPO_ROOT}/notebooks/xDESI/abacus_paste/stage31_pz3_cap2400_hmcbestfit_mmin11p147538_nside2048_lmax4096.selected.yaml}"
CHECKPOINT_PASTE_RUN_ROOT_BASE="${CHECKPOINT_PASTE_RUN_ROOT_BASE:-${REPO_ROOT}/data/xDESI/processed/abacus_backlight/stage31_pz3_cap2400_hmccheckpoints_mmin11p147538}"
CHECKPOINT_PASTE_NSIDE="${CHECKPOINT_PASTE_NSIDE:-2048}"
CHECKPOINT_PASTE_LMAX="${CHECKPOINT_PASTE_LMAX:-4096}"
CHECKPOINT_PASTE_NUM_SPLITS="${CHECKPOINT_PASTE_NUM_SPLITS:-4}"
CHECKPOINT_PASTE_PIXEL_WORKERS="${CHECKPOINT_PASTE_PIXEL_WORKERS:-16}"
CHECKPOINT_PASTE_CATALOG_SOURCE="${CHECKPOINT_PASTE_CATALOG_SOURCE:-${REPO_ROOT}/data/xDESI/processed/abacus_backlight/stage31_pz3_cap2400_hmcfailed_mmin11p147538/halos/abacus_c9999_ph9999_pz3cap2400_hmcfailed_z0p63_0p98_logMgt11p147538_halos.h5}"
CHECKPOINT_PASTE_CATALOG_OUTPUT_NAME="${CHECKPOINT_PASTE_CATALOG_OUTPUT_NAME:-abacus_c9999_ph9999_pz3cap2400_hmcbestfit_z0p63_0p98_logMgt11p147538_halos.h5}"
if [[ -z "${CHECKPOINT_PASTE_DO_PREPROCESS+x}" ]]; then
  if [[ -f "${CHECKPOINT_PASTE_CATALOG_SOURCE}" ]]; then
    CHECKPOINT_PASTE_DO_PREPROCESS=0
  else
    CHECKPOINT_PASTE_DO_PREPROCESS=1
  fi
fi
CHECKPOINT_PASTE_DO_PASTED_THEORY="${CHECKPOINT_PASTE_DO_PASTED_THEORY:-1}"
CHECKPOINT_PASTE_DO_DIRECT_FIELD="${CHECKPOINT_PASTE_DO_DIRECT_FIELD:-0}"
CHECKPOINT_PASTE_DO_PLUS_DIRECT="${CHECKPOINT_PASTE_DO_PLUS_DIRECT:-0}"
CHECKPOINT_PASTE_SIM_MATCHED_TRANSFERS="${CHECKPOINT_PASTE_SIM_MATCHED_TRANSFERS:-1}"
CHECKPOINT_POSTPROCESS_PLATFORM="${CHECKPOINT_POSTPROCESS_PLATFORM:-cpu}"
KSZ_VELOCITY_MODE="${KSZ_VELOCITY_MODE:-photoz_reconstruction_emulation}"
KSZ_RECONSTRUCTION_NOISE_SEED="${KSZ_RECONSTRUCTION_NOISE_SEED:-12345}"
KSZ_YLIM_MIN="${KSZ_YLIM_MIN:--5e-5}"
KSZ_YLIM_MAX="${KSZ_YLIM_MAX:-5e-5}"
PLOT_ELL_MAX="${PLOT_ELL_MAX:-2800}"

VALIDATE_ONLY="${VALIDATE_ONLY:-0}"
COMPARE_FIDUCIAL="${COMPARE_FIDUCIAL:-0}"
DEBUG_INIT="${DEBUG_INIT:-0}"
if [[ "${VALIDATE_ONLY}" == "1" ]]; then
  COMBINE_AFTER=0
  CHECKPOINT_SAMPLES_EVERY=0
  CHECKPOINT_COMBINE_AFTER=0
fi

DEFAULT_INIT_PARAMS="${REPO_ROOT}/notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_midres2048_true_nz_hmc_stage31_multigpu/stage31_hmc_abacus_cosmo_midres2048_simple1h2h_lmax3000_gk1000_mmin11p147538_60param_depth6_defaultacc_warm25_2000x16_checkpoint25_v1/combined/checkpoints/checkpoint_001250/bestfit_params_stage31_multigpu_abacus_cosmo_midres2048_simple1h2h_lmax3000_gk1000_mmin11p147538_60param_depth6_defaultacc_warm25_2000x16_checkpoint25_v1_checkpoint_001250.yaml"
if [[ ! -f "${DEFAULT_INIT_PARAMS}" ]]; then
  DEFAULT_INIT_PARAMS="none"
fi
INIT_PARAMS="${INIT_PARAMS:-${DEFAULT_INIT_PARAMS}}"
if [[ "${INIT_PARAMS}" == "none" ]]; then
  INIT_PARAMS=""
fi

RUN_BASE_DIR="${RUN_BASE_DIR:-${REPO_ROOT}/notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_midres2048_true_nz_hmc_stage31_multigpu}"
RUN_DIR="${RUN_BASE_DIR}/${RUN_LABEL}"
WORKER_ROOT="${RUN_DIR}/workers"
COMBINED_DIR="${RUN_DIR}/combined"
WORKER_LOG_DIR="${RUN_DIR}/worker_logs"

mkdir -p "${LOG_DIR}" "${WORKER_ROOT}" "${COMBINED_DIR}" "${WORKER_LOG_DIR}"
cd "${REPO_ROOT}"

export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${RUN_DIR}/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

if [[ -n "${CRAY_CUDATOOLKIT_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${CRAY_CUDATOOLKIT_DIR}/lib64:${LD_LIBRARY_PATH:-}"
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_cuda_data_dir=${CRAY_CUDATOOLKIT_DIR}"
fi

echo "[$(date)] host=$(hostname)"
echo "[$(date)] config=${CONFIG}"
echo "[$(date)] run_version=${RUN_VERSION}"
echo "[$(date)] run_label=${RUN_LABEL}"
echo "[$(date)] combined_suffix=${COMBINED_SUFFIX}"
echo "[$(date)] run_base_dir=${RUN_BASE_DIR}"
echo "[$(date)] run_dir=${RUN_DIR}"
echo "[$(date)] n_workers=${N_WORKERS}"
echo "[$(date)] chains_per_gpu=${CHAINS_PER_GPU}"
echo "[$(date)] total_chains=$((N_WORKERS * CHAINS_PER_GPU))"
echo "[$(date)] num_warmup=${NUM_WARMUP}"
echo "[$(date)] num_samples=${NUM_SAMPLES}"
echo "[$(date)] max_tree_depth=${MAX_TREE_DEPTH}"
echo "[$(date)] target_accept=${TARGET_ACCEPT}"
echo "[$(date)] heartbeat_seconds=${HEARTBEAT_SECONDS}"
echo "[$(date)] checkpoint_samples_every=${CHECKPOINT_SAMPLES_EVERY}"
echo "[$(date)] checkpoint_combine_after=${CHECKPOINT_COMBINE_AFTER}"
echo "[$(date)] checkpoint_paste_after=${CHECKPOINT_PASTE_AFTER}"
echo "[$(date)] checkpoint_monitor_interval=${CHECKPOINT_MONITOR_INTERVAL}"
echo "[$(date)] checkpoint_paste_nside=${CHECKPOINT_PASTE_NSIDE}"
echo "[$(date)] checkpoint_paste_lmax=${CHECKPOINT_PASTE_LMAX}"
echo "[$(date)] checkpoint_paste_run_root_base=${CHECKPOINT_PASTE_RUN_ROOT_BASE}"
echo "[$(date)] checkpoint_paste_catalog_source=${CHECKPOINT_PASTE_CATALOG_SOURCE}"
echo "[$(date)] checkpoint_paste_catalog_output_name=${CHECKPOINT_PASTE_CATALOG_OUTPUT_NAME}"
echo "[$(date)] checkpoint_paste_do_preprocess=${CHECKPOINT_PASTE_DO_PREPROCESS}"
echo "[$(date)] checkpoint_postprocess_platform=${CHECKPOINT_POSTPROCESS_PLATFORM}"
echo "[$(date)] plot_ell_max=${PLOT_ELL_MAX}"
echo "[$(date)] init_params=${INIT_PARAMS:-fiducial}"
echo "[$(date)] validate_only=${VALIDATE_ONLY}"
echo "[$(date)] compare_fiducial=${COMPARE_FIDUCIAL}"
echo "[$(date)] debug_init=${DEBUG_INIT}"
echo "[$(date)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

if [[ -n "${INIT_PARAMS}" && ! -f "${INIT_PARAMS}" ]]; then
  echo "Requested INIT_PARAMS does not exist: ${INIT_PARAMS}" >&2
  exit 2
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
else
  ALLOCATED_GPUS=(0 1 2 3)
fi
if (( N_WORKERS > ${#ALLOCATED_GPUS[@]} )); then
  echo "N_WORKERS=${N_WORKERS} exceeds visible GPU count=${#ALLOCATED_GPUS[@]} (${ALLOCATED_GPUS[*]})" >&2
  exit 2
fi
echo "[$(date)] allocated_gpu_slots=${ALLOCATED_GPUS[*]}"

declare -a PIDS=()
for rank in $(seq 0 $((N_WORKERS - 1))); do
  seed=$((BASE_SEED + rank))
  worker_dir="${WORKER_ROOT}/worker_${rank}"
  mkdir -p "${worker_dir}"
  worker_log="${WORKER_LOG_DIR}/worker_${rank}.out"
  worker_err="${WORKER_LOG_DIR}/worker_${rank}.err"
  gpu_id="${ALLOCATED_GPUS[$rank]}"

  echo "[$(date)] launching worker ${rank}, gpu=${gpu_id}, seed=${seed}, output=${worker_dir}"
  (
    export REPO_ROOT="${REPO_ROOT}"
    export PYTHON="${PYTHON}"
    export RUNNER="${RUNNER}"
    export CONFIG="${CONFIG}"
    export WORKER_RANK="${rank}"
    export WORKER_SEED="${seed}"
    export WORKER_DIR="${worker_dir}"
    export NUM_WARMUP="${NUM_WARMUP}"
    export NUM_SAMPLES="${NUM_SAMPLES}"
    export CHAINS_PER_GPU="${CHAINS_PER_GPU}"
    export MAX_TREE_DEPTH="${MAX_TREE_DEPTH}"
    export TARGET_ACCEPT="${TARGET_ACCEPT}"
    export INIT_PARAMS="${INIT_PARAMS}"
    export GPU_SANITY_CHECK="${GPU_SANITY_CHECK}"
    export HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS}"
    export CHECKPOINT_SAMPLES_EVERY="${CHECKPOINT_SAMPLES_EVERY}"
    export JAX_PLATFORMS="${JAX_PLATFORMS}"
    export JAX_ENABLE_X64="${JAX_ENABLE_X64}"
    export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE}"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION}"
    export PYTHONUNBUFFERED="${PYTHONUNBUFFERED}"
    export MPLCONFIGDIR="${MPLCONFIGDIR}"
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export JAX_VISIBLE_DEVICES=0
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-15}"
    bash -c '
        set -euo pipefail
        cd "${REPO_ROOT}"
        echo "[$(date)] worker=${WORKER_RANK} host=$(hostname)"
        echo "[$(date)] worker=${WORKER_RANK} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
        echo "[$(date)] worker=${WORKER_RANK} JAX_VISIBLE_DEVICES=${JAX_VISIBLE_DEVICES:-unset}"
        nvidia-smi || true
        heartbeat_pid=""
        if [[ "${HEARTBEAT_SECONDS}" != "0" ]]; then
          (
            while true; do
              echo "[$(date)] worker=${WORKER_RANK} heartbeat"
              nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits || true
              sleep "${HEARTBEAT_SECONDS}"
            done
          ) &
          heartbeat_pid="$!"
          cleanup_heartbeat() {
            if [[ -n "${heartbeat_pid}" ]]; then
              kill "${heartbeat_pid}" 2>/dev/null || true
              wait "${heartbeat_pid}" 2>/dev/null || true
            fi
          }
          trap cleanup_heartbeat EXIT
        fi
        args=(
          "${RUNNER}"
          --config "${CONFIG}"
          --platform gpu
          --num-warmup "${NUM_WARMUP}"
          --num-samples "${NUM_SAMPLES}"
          --num-chains "${CHAINS_PER_GPU}"
          --chain-method vectorized
          --max-tree-depth "${MAX_TREE_DEPTH}"
          --seed "${WORKER_SEED}"
          --output-dir "${WORKER_DIR}"
          --no-progress
        )
        if [[ -n "${TARGET_ACCEPT}" ]]; then
          args+=(--target-accept-prob "${TARGET_ACCEPT}")
        fi
        if [[ "${GPU_SANITY_CHECK}" == "1" ]]; then
          args+=(--gpu-sanity-check --gpu-sanity-matrix-size 2048)
        fi
        if [[ "${VALIDATE_ONLY}" == "1" ]]; then
          args+=(--validate-only)
        fi
        if [[ "${COMPARE_FIDUCIAL}" == "1" ]]; then
          args+=(--compare-fiducial)
        fi
        if [[ "${DEBUG_INIT}" == "1" ]]; then
          args+=(--debug-init)
        fi
        if [[ "${CHECKPOINT_SAMPLES_EVERY}" != "0" ]]; then
          args+=(--checkpoint-samples-every "${CHECKPOINT_SAMPLES_EVERY}")
        fi
        if [[ -n "${INIT_PARAMS}" ]]; then
          args+=(--init-params "${INIT_PARAMS}")
        fi
        echo "[$(date)] worker=${WORKER_RANK} starting HMC command: ${PYTHON} -u ${args[*]}"
        "${PYTHON}" -u "${args[@]}"
        echo "[$(date)] worker=${WORKER_RANK} HMC command finished"
      '
  ) >"${worker_log}" 2>"${worker_err}" &
  PIDS+=("$!")
done

CHECKPOINT_MONITOR_PID=""
CHECKPOINT_MONITOR_STOP_FILE="${RUN_DIR}/checkpoint_monitor.stop"
rm -f "${CHECKPOINT_MONITOR_STOP_FILE}"
if [[ "${CHECKPOINT_SAMPLES_EVERY}" != "0" && "${CHECKPOINT_COMBINE_AFTER}" == "1" ]]; then
  monitor_args=(
    "${CHECKPOINT_MONITOR}"
    --config "${CONFIG}"
    --worker-dir "${WORKER_ROOT}"
    --combined-dir "${COMBINED_DIR}"
    --combined-suffix "${COMBINED_SUFFIX}"
    --run-label "${RUN_LABEL}"
    --expected-workers "${N_WORKERS}"
    --poll-interval "${CHECKPOINT_MONITOR_INTERVAL}"
    --stop-file "${CHECKPOINT_MONITOR_STOP_FILE}"
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
    --postprocess-platform "${CHECKPOINT_POSTPROCESS_PLATFORM}"
    --retry-failed
  )
  if [[ "${CHECKPOINT_PASTE_AFTER}" == "1" ]]; then
    monitor_args+=(--submit-paste)
  fi
  echo "[$(date)] starting checkpoint monitor: ${PYTHON} -u ${monitor_args[*]}"
  "${PYTHON}" -u "${monitor_args[@]}" >"${WORKER_LOG_DIR}/checkpoint_monitor.out" 2>"${WORKER_LOG_DIR}/checkpoint_monitor.err" &
  CHECKPOINT_MONITOR_PID="$!"
fi

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

monitor_status=0
if [[ -n "${CHECKPOINT_MONITOR_PID}" ]]; then
  echo "[$(date)] stopping checkpoint monitor"
  touch "${CHECKPOINT_MONITOR_STOP_FILE}"
  if ! wait "${CHECKPOINT_MONITOR_PID}"; then
    monitor_status=1
  fi
fi

if [[ "${status}" != "0" ]]; then
  echo "[$(date)] at least one worker failed. See ${WORKER_LOG_DIR}" >&2
  exit "${status}"
fi
if [[ "${monitor_status}" != "0" ]]; then
  echo "[$(date)] checkpoint monitor failed. See ${WORKER_LOG_DIR}/checkpoint_monitor.err" >&2
  exit "${monitor_status}"
fi

echo "[$(date)] all workers finished"

if [[ "${COMBINE_AFTER}" == "1" ]]; then
  echo "[$(date)] combining worker chains"
  CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[0]}" JAX_VISIBLE_DEVICES=0 "${PYTHON}" -u "${COMBINER}" \
    --config "${CONFIG}" \
    --worker-dir "${WORKER_ROOT}" \
    --output-dir "${COMBINED_DIR}" \
    --suffix "${COMBINED_SUFFIX}" \
    --plot-ell-max "${PLOT_ELL_MAX}" \
    --plot-ksz-ylim="${KSZ_YLIM_MIN},${KSZ_YLIM_MAX}"
fi

echo "[$(date)] done"
