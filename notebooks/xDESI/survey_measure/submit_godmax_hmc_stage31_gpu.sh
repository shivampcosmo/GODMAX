#!/bin/bash
#SBATCH -p gpu
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=15
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --job-name=xdesi_hmc_stage31
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%j.err

set -euo pipefail

MODE="${1:-smoke}"
REPO_ROOT="/mnt/ceph/users/spandey/ltu-godmax/GODMAX"
PYTHON="/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python"
RUNNER="${REPO_ROOT}/notebooks/xDESI/survey_measure/run_godmax_multiprobe_hmc_stage31.py"
LOG_DIR="${REPO_ROOT}/notebooks/xDESI/survey_measure/logs"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

# Force JAX/NumPyro onto CUDA and make failures loud if no GPU backend is present.
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True

# Reserve 95% of the assigned GPU memory up front. This must be set before
# Python imports JAX.
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# Local GPU jobs in this repo commonly need the CUDA data dir on Cray systems.
if [[ -n "${CRAY_CUDATOOLKIT_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${CRAY_CUDATOOLKIT_DIR}/lib64:${LD_LIBRARY_PATH:-}"
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_cuda_data_dir=${CRAY_CUDATOOLKIT_DIR}"
fi

echo "[$(date)] host=$(hostname)"
echo "[$(date)] mode=${MODE}"
echo "[$(date)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[$(date)] JAX_PLATFORMS=${JAX_PLATFORMS}"
echo "[$(date)] XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE}"
echo "[$(date)] XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION}"
nvidia-smi || true

run_smoke() {
  "${PYTHON}" "${RUNNER}" \
    --platform gpu \
    --debug-init \
    --smoke
}

run_full() {
  "${PYTHON}" "${RUNNER}" \
    --platform gpu
}

case "${MODE}" in
  smoke)
    run_smoke
    ;;
  full)
    run_full
    ;;
  both)
    run_smoke
    run_full
    ;;
  validate)
    "${PYTHON}" "${RUNNER}" --platform gpu --validate-only --compare-fiducial --debug-init
    ;;
  *)
    echo "Unknown mode '${MODE}'. Use: smoke, full, both, or validate." >&2
    exit 2
    ;;
esac

echo "[$(date)] done"
