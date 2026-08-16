#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=genx
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=xdesi_highres4096_finalize
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%j.err

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 PLAN_PATH OUTPUT_DIR MANIFEST_PATH" >&2
  exit 2
fi

PLAN_PATH="$1"
OUTPUT_DIR="$2"
MANIFEST_PATH="$3"
REPO_ROOT="/mnt/ceph/users/spandey/ltu-godmax/GODMAX"
PYTHON="/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python"
DRIVER="${REPO_ROOT}/notebooks/xDESI/survey_measure/run_multiprobe_production.py"

IFS=: read -r -a runtime_source_files <<< "${XDESI_RUNTIME_SOURCE_FILES:?missing runtime source file list}"
actual_runtime_sha256="$(sha256sum "${runtime_source_files[@]}" | sha256sum | cut -d' ' -f1)"
if [[ "${actual_runtime_sha256}" != "${XDESI_RUNTIME_SOURCE_SHA256:?missing frozen runtime digest}" ]]; then
  echo "Runtime source changed after submission: expected ${XDESI_RUNTIME_SOURCE_SHA256}, got ${actual_runtime_sha256}." >&2
  exit 3
fi
actual_plan_sha256="$(sha256sum "${PLAN_PATH}" | cut -d' ' -f1)"
if [[ "${actual_plan_sha256}" != "${XDESI_COV_WORK_PLAN_SHA256:?missing frozen work-plan digest}" ]]; then
  echo "Covariance work plan changed after submission: expected ${XDESI_COV_WORK_PLAN_SHA256}, got ${actual_plan_sha256}." >&2
  exit 4
fi

module purge
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-xdesi-${USER}-${SLURM_JOB_ID:-local}}"

echo "Runtime source SHA256: ${actual_runtime_sha256}"
echo "Covariance work-plan file SHA256: ${actual_plan_sha256}"
echo "Finalizing highres4096 measurement in ${OUTPUT_DIR}"

cd "${REPO_ROOT}"
srun --exact --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-2}" \
  --cpu-bind=cores \
  "${PYTHON}" "${DRIVER}" finalize \
    --stage highres4096 \
    --output-dir "${OUTPUT_DIR}" \
    --manifest-path "${MANIFEST_PATH}" \
    --plan-path "${PLAN_PATH}" \
    --plot-ell-max 8192 \
    --plot-ksz-scale 1000 \
    --plot-ksz-ylim auto
