#!/bin/bash
#SBATCH -p gpu
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=15
#SBATCH --mem=800G
#SBATCH --time=18:00:00
#SBATCH --job-name=xdesi_map_stage31_4gpu
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%j.err

set -euo pipefail

REPO_ROOT="/mnt/ceph/users/spandey/ltu-godmax/GODMAX"
PYTHON="/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python"
OPTIMIZER="${REPO_ROOT}/notebooks/xDESI/survey_measure/optimize_godmax_multiprobe_stage31.py"
CONFIG="${REPO_ROOT}/param_files/xDESI/params_multiprobe_fast1024_hmc_stage31.yaml"
LOG_DIR="${REPO_ROOT}/notebooks/xDESI/survey_measure/logs"

RUN_LABEL="${1:-stage31_map_${SLURM_JOB_ID:-manual}}"
N_WORKERS="${N_WORKERS:-4}"
STARTS_PER_GPU="${STARTS_PER_GPU:-2}"
METHOD="${METHOD:-adam-lbfgsb}"
ADAM_STEPS="${ADAM_STEPS:-60}"
ADAM_LR="${ADAM_LR:-0.002}"
ADAM_GRAD_CLIP="${ADAM_GRAD_CLIP:-1000000}"
LBFGS_MAXITER="${LBFGS_MAXITER:-80}"
LBFGS_MAXFUN="${LBFGS_MAXFUN:-120}"
LOG_EVERY="${LOG_EVERY:-1}"
EVAL_LOG_EVERY="${EVAL_LOG_EVERY:-1}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-120}"
TARGET_RANDOM_MODE="${TARGET_RANDOM_MODE:-around-init}"
START_JITTER="${START_JITTER:-0.08}"
BASE_SEED="${BASE_SEED:-73000}"
GPU_SANITY_CHECK="${GPU_SANITY_CHECK:-1}"
COMBINE_AFTER="${COMBINE_AFTER:-1}"

DEFAULT_INIT_PARAMS="${REPO_ROOT}/notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_fast1024_true_nz_hmc_stage31_local/bestfit_params_smoke_stage31.yaml"
INIT_PARAMS="${INIT_PARAMS:-${DEFAULT_INIT_PARAMS}}"
if [[ "${INIT_PARAMS}" == "none" ]]; then
  INIT_PARAMS=""
fi

RUN_DIR="${REPO_ROOT}/notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_fast1024_true_nz_map_stage31_multigpu/${RUN_LABEL}"
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
echo "[$(date)] run_label=${RUN_LABEL}"
echo "[$(date)] run_dir=${RUN_DIR}"
echo "[$(date)] method=${METHOD}"
echo "[$(date)] n_workers=${N_WORKERS}"
echo "[$(date)] starts_per_gpu=${STARTS_PER_GPU}"
echo "[$(date)] adam_steps=${ADAM_STEPS}"
echo "[$(date)] adam_lr=${ADAM_LR}"
echo "[$(date)] lbfgs_maxiter=${LBFGS_MAXITER}"
echo "[$(date)] lbfgs_maxfun=${LBFGS_MAXFUN}"
echo "[$(date)] log_every=${LOG_EVERY}"
echo "[$(date)] eval_log_every=${EVAL_LOG_EVERY}"
echo "[$(date)] heartbeat_seconds=${HEARTBEAT_SECONDS}"
echo "[$(date)] random_mode=${TARGET_RANDOM_MODE}"
echo "[$(date)] start_jitter=${START_JITTER}"
echo "[$(date)] init_params=${INIT_PARAMS:-fiducial/random only}"
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

  echo "[$(date)] launching MAP worker ${rank}, gpu=${gpu_id}, seed=${seed}, output=${worker_dir}"
  (
    export REPO_ROOT="${REPO_ROOT}"
    export PYTHON="${PYTHON}"
    export OPTIMIZER="${OPTIMIZER}"
    export CONFIG="${CONFIG}"
    export WORKER_RANK="${rank}"
    export WORKER_SEED="${seed}"
    export WORKER_DIR="${worker_dir}"
    export STARTS_PER_GPU="${STARTS_PER_GPU}"
    export METHOD="${METHOD}"
    export ADAM_STEPS="${ADAM_STEPS}"
    export ADAM_LR="${ADAM_LR}"
    export ADAM_GRAD_CLIP="${ADAM_GRAD_CLIP}"
    export LBFGS_MAXITER="${LBFGS_MAXITER}"
    export LBFGS_MAXFUN="${LBFGS_MAXFUN}"
    export LOG_EVERY="${LOG_EVERY}"
    export EVAL_LOG_EVERY="${EVAL_LOG_EVERY}"
    export HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS}"
    export TARGET_RANDOM_MODE="${TARGET_RANDOM_MODE}"
    export START_JITTER="${START_JITTER}"
    export INIT_PARAMS="${INIT_PARAMS}"
    export GPU_SANITY_CHECK="${GPU_SANITY_CHECK}"
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
          "${OPTIMIZER}"
          --config "${CONFIG}"
          --platform gpu
          --output-dir "${WORKER_DIR}"
          --suffix "stage31_map_worker_${WORKER_RANK}"
          --method "${METHOD}"
          --num-starts "${STARTS_PER_GPU}"
          --seed "${WORKER_SEED}"
          --random-mode "${TARGET_RANDOM_MODE}"
          --start-jitter "${START_JITTER}"
          --adam-steps "${ADAM_STEPS}"
          --adam-lr "${ADAM_LR}"
          --adam-grad-clip "${ADAM_GRAD_CLIP}"
          --lbfgs-maxiter "${LBFGS_MAXITER}"
          --lbfgs-maxfun "${LBFGS_MAXFUN}"
          --log-every "${LOG_EVERY}"
          --eval-log-every "${EVAL_LOG_EVERY}"
        )
        if [[ "${GPU_SANITY_CHECK}" == "1" ]]; then
          args+=(--gpu-sanity-check --gpu-sanity-matrix-size 2048)
        fi
        if [[ -n "${INIT_PARAMS}" ]]; then
          args+=(--init-params "${INIT_PARAMS}")
        fi
        if [[ "${WORKER_RANK}" != "0" ]]; then
          args+=(--no-init-start --no-fiducial-start)
        fi
        "${PYTHON}" -u "${args[@]}"
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
  echo "[$(date)] at least one MAP worker failed. See ${WORKER_LOG_DIR}" >&2
  exit "${status}"
fi

echo "[$(date)] all MAP workers finished"

if [[ "${COMBINE_AFTER}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[0]}" JAX_VISIBLE_DEVICES=0 "${PYTHON}" "${OPTIMIZER}" \
    --config "${CONFIG}" \
    --platform gpu \
    --output-dir "${COMBINED_DIR}" \
    --suffix stage31_map_multigpu \
    --combine-worker-dir "${WORKER_ROOT}" \
    --gpu-sanity-check \
    --gpu-sanity-matrix-size 2048
fi

echo "[$(date)] done"
