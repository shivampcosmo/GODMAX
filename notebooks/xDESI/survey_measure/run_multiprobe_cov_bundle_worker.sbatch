#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=11
#SBATCH --cpus-per-task=11
#SBATCH --constraint=rome
#SBATCH --partition=cmbas
#SBATCH --mem=880G
#SBATCH --time=04:00:00
#SBATCH --job-name=xdesi_highres4096_cov_bundle
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%A_%a.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%A_%a.err

set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 PLAN_PATH OUTPUT_DIR MANIFEST_PATH MAPS_PATH" >&2
  exit 2
fi

PLAN_PATH="$1"
OUTPUT_DIR="$2"
MANIFEST_PATH="$3"
SOURCE_MAP="$4"
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

bundle_id="${SLURM_ARRAY_TASK_ID:-${XDESI_COV_BUNDLE_ID:-0}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-xdesi-${USER}-${SLURM_JOB_ID:-local}-${bundle_id}}"
group_line="$(${PYTHON} "${DRIVER}" show-cov-work-bundle \
  --stage highres4096 \
  --output-dir "${OUTPUT_DIR}" \
  --manifest-path "${MANIFEST_PATH}" \
  --plan-path "${PLAN_PATH}" \
  --batch-id "${bundle_id}")"
read -r -a group_indices <<< "${group_line}"
if [[ "${#group_indices[@]}" -eq 0 ]]; then
  echo "Bundle ${bundle_id} contains no covariance groups." >&2
  exit 5
fi
if [[ "${#group_indices[@]}" -gt "${SLURM_NTASKS:-11}" ]]; then
  echo "Bundle ${bundle_id} has ${#group_indices[@]} groups but allocation has ${SLURM_NTASKS:-11} tasks." >&2
  exit 6
fi

module purge
export OMP_NUM_THREADS=11
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

scratch_parent="${SLURM_TMPDIR:-/tmp}"
scratch_dir="${scratch_parent}/xdesi-cov-${SLURM_JOB_ID:-local}-${bundle_id}"
mkdir -p "${scratch_dir}"
local_map="${scratch_dir}/$(basename "${SOURCE_MAP}")"

pids=()
cleanup() {
  rm -f -- "${local_map}" "${local_map}.part"
  rmdir -- "${scratch_dir}" 2>/dev/null || true
}
terminate_children() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'terminate_children; exit 130' INT TERM

echo "Runtime source SHA256: ${actual_runtime_sha256}"
echo "Covariance work-plan file SHA256: ${actual_plan_sha256}"
echo "Bundle ${bundle_id}: groups ${group_indices[*]}"
echo "Allocation: nodes=${SLURM_JOB_NUM_NODES:-1} tasks=${SLURM_NTASKS:-11} cpus_per_task=${SLURM_CPUS_PER_TASK:-11}"
echo "Staging map once to node-local storage: ${SOURCE_MAP} -> ${local_map}"
cp -- "${SOURCE_MAP}" "${local_map}.part"
source_size="$(stat -c '%s' "${SOURCE_MAP}")"
local_size="$(stat -c '%s' "${local_map}.part")"
if [[ "${source_size}" != "${local_size}" ]]; then
  echo "Node-local map copy has size ${local_size}, expected ${source_size}." >&2
  exit 7
fi
mv -- "${local_map}.part" "${local_map}"

cd "${REPO_ROOT}"
for group_index in "${group_indices[@]}"; do
  echo "Launching global covariance group ${group_index} with 11 cores and 80 GiB."
  srun --exclusive --exact \
    --nodes=1 --ntasks=1 --cpus-per-task=11 \
    --mem=80G --cpu-bind=cores --kill-on-bad-exit=1 --label \
    "${PYTHON}" "${DRIVER}" cov-key \
      --stage highres4096 \
      --output-dir "${OUTPUT_DIR}" \
      --maps-path "${local_map}" \
      --manifest-path "${MANIFEST_PATH}" \
      --cov-class all \
      --task-id "${group_index}" \
      --no-cov-workspace-cache &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "At least one covariance group in bundle ${bundle_id} failed." >&2
  exit "${status}"
fi
echo "Bundle ${bundle_id} completed all ${#group_indices[@]} groups."
