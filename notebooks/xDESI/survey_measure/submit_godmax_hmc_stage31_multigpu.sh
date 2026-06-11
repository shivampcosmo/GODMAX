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
CONFIG="${CONFIG:-${REPO_ROOT}/param_files/xDESI/params_multiprobe_fast1024_hmc_stage31.yaml}"
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

RUN_VERSION="${RUN_VERSION:-mmin11p147538_v1}"
RUN_LABEL="${1:-stage31_hmc_${RUN_VERSION}}"
COMBINED_SUFFIX="${COMBINED_SUFFIX:-stage31_multigpu_${RUN_VERSION}}"
N_WORKERS="${N_WORKERS:-4}"
CHAINS_PER_GPU="${CHAINS_PER_GPU:-$(yaml_value "${CONFIG}" sampler.num_chains)}"
NUM_WARMUP="${NUM_WARMUP:-$(yaml_value "${CONFIG}" sampler.num_warmup)}"
NUM_SAMPLES="${NUM_SAMPLES:-$(yaml_value "${CONFIG}" sampler.num_samples)}"
MAX_TREE_DEPTH="${MAX_TREE_DEPTH:-$(yaml_value "${CONFIG}" sampler.max_tree_depth)}"
TARGET_ACCEPT="${TARGET_ACCEPT:-$(yaml_value "${CONFIG}" sampler.target_accept_prob)}"
BASE_SEED="${BASE_SEED:-42000}"
GPU_SANITY_CHECK="${GPU_SANITY_CHECK:-1}"
COMBINE_AFTER="${COMBINE_AFTER:-1}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-120}"

DEFAULT_INIT_PARAMS="${REPO_ROOT}/param_files/xDESI/params_multiprobe_fast1024_true_nz_stage31_bestfit_v2.yaml"
if [[ ! -f "${DEFAULT_INIT_PARAMS}" ]]; then
  DEFAULT_INIT_PARAMS="${REPO_ROOT}/param_files/xDESI/params_multiprobe_fast1024_true_nz_stage31_bestfit_v1.yaml"
fi
INIT_PARAMS="${INIT_PARAMS:-${DEFAULT_INIT_PARAMS}}"
if [[ "${INIT_PARAMS}" == "none" ]]; then
  INIT_PARAMS=""
fi

RUN_BASE_DIR="${RUN_BASE_DIR:-${REPO_ROOT}/notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_fast1024_true_nz_hmc_stage31_multigpu}"
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
echo "[$(date)] init_params=${INIT_PARAMS:-fiducial}"
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
          --target-accept-prob "${TARGET_ACCEPT}"
          --seed "${WORKER_SEED}"
          --output-dir "${WORKER_DIR}"
          --no-progress
        )
        if [[ "${GPU_SANITY_CHECK}" == "1" ]]; then
          args+=(--gpu-sanity-check --gpu-sanity-matrix-size 2048)
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

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" != "0" ]]; then
  echo "[$(date)] at least one worker failed. See ${WORKER_LOG_DIR}" >&2
  exit "${status}"
fi

echo "[$(date)] all workers finished"

if [[ "${COMBINE_AFTER}" == "1" ]]; then
  echo "[$(date)] combining worker chains"
  CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[0]}" JAX_VISIBLE_DEVICES=0 "${PYTHON}" -u "${COMBINER}" \
    --config "${CONFIG}" \
    --worker-dir "${WORKER_ROOT}" \
    --output-dir "${COMBINED_DIR}" \
    --suffix "${COMBINED_SUFFIX}"
fi

echo "[$(date)] done"
