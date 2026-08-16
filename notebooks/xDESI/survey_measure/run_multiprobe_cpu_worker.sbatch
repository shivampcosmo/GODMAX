#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -C rome
#SBATCH -p cmbas
#SBATCH --time=12:00:00
#SBATCH --job-name=xdesi_multiprobe
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%A_%a.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%A_%a.err

set -euo pipefail

REPO_ROOT="/mnt/ceph/users/spandey/ltu-godmax/GODMAX"
PYTHON="/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python"
DRIVER="${REPO_ROOT}/notebooks/xDESI/survey_measure/run_multiprobe_production.py"

if [[ -n "${XDESI_RUNTIME_SOURCE_SHA256:-}" ]]; then
  IFS=: read -r -a runtime_source_files <<< "${XDESI_RUNTIME_SOURCE_FILES:?missing runtime source file list}"
  actual_runtime_sha256="$(sha256sum "${runtime_source_files[@]}" | sha256sum | cut -d' ' -f1)"
  if [[ "${actual_runtime_sha256}" != "${XDESI_RUNTIME_SOURCE_SHA256}" ]]; then
    echo "Runtime source changed after submission: expected ${XDESI_RUNTIME_SOURCE_SHA256}, got ${actual_runtime_sha256}." >&2
    exit 3
  fi
  echo "Runtime source SHA256: ${actual_runtime_sha256}"
fi

echo "--- JOB DETAILS ---"
echo "Job Name: ${SLURM_JOB_NAME:-unknown} (${SLURM_JOB_ID:-no_job_id})"
echo "Array Task: ${SLURM_ARRAY_TASK_ID:-none}"
echo "Running on: $(hostname)"
echo "Command: ${DRIVER} $*"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-1}"
echo "-------------------"

module purge

export OMP_NUM_THREADS="${XDESI_OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-xdesi-${USER}}"

cd "${REPO_ROOT}"
srun --cpu-bind=cores "${PYTHON}" "${DRIVER}" "$@"
