#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH -C rome
#SBATCH -p cmbas
#SBATCH --time=08:00:00
#SBATCH --job-name=xdesi_dr9_random8
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%j.err

set -euo pipefail

REPO_ROOT="/mnt/ceph/users/spandey/ltu-godmax/GODMAX"
PYTHON="/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python"
BUILDER="${REPO_ROOT}/measurements/xDESI/scripts/prepare_desi_dr9_extended_catalogs.py"
TRANSFER_ROOT="${REPO_ROOT}/data/xDESI/survey_data"
SOURCE_ROOT="${TRANSFER_ROOT}/desi_dr9/legacy-survey-0.49.0"

if [[ -n "${XDESI_RANDOM_RUNTIME_SOURCE_SHA256:-}" ]]; then
  actual_sha256="$(sha256sum "${BUILDER}" | cut -d' ' -f1)"
  if [[ "${actual_sha256}" != "${XDESI_RANDOM_RUNTIME_SOURCE_SHA256}" ]]; then
    echo "Random-mask builder changed after submission: expected ${XDESI_RANDOM_RUNTIME_SOURCE_SHA256}, got ${actual_sha256}." >&2
    exit 3
  fi
fi

module purge
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-xdesi-${USER}}"

cd "${REPO_ROOT}"
common=(
  --transfer-root "${TRANSFER_ROOT}"
  --random-source-root "${SOURCE_ROOT}"
  --random-indices 0 1 10 11 12 13 14 15
  --skip-catalogs
  --random-chunk-size 1000000
)

# Fast schema/alignment gate, then the full streamed build.  Re-running the same
# command once more exercises the identity/schema reuse validator without writes.
srun --cpu-bind=cores "${PYTHON}" "${BUILDER}" "${common[@]}" --validate-only
srun --cpu-bind=cores "${PYTHON}" "${BUILDER}" "${common[@]}" --verify-full-sha256
srun --cpu-bind=cores "${PYTHON}" "${BUILDER}" "${common[@]}"
