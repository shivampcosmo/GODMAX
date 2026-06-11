#!/bin/bash
set -euo pipefail

REPO_ROOT="/mnt/ceph/users/spandey/ltu-godmax/GODMAX"
PYTHON="/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python"
DRIVER="${REPO_ROOT}/notebooks/xDESI/survey_measure/run_multiprobe_production.py"
WORKER="${REPO_ROOT}/notebooks/xDESI/survey_measure/run_multiprobe_cpu_worker.slurm"
LOG_DIR="${REPO_ROOT}/notebooks/xDESI/survey_measure/logs"
OUTPUT_DIR="${OUTPUT_DIR:-data/xDESI/processed/multiprobe_namaster}"
STAGES="${STAGES:-fast1024}"
GATE_MIDRES_ON_FAST="${GATE_MIDRES_ON_FAST:-0}"
FORCE_FLAG=""
FAST_COV_SCALAR_ARRAY_CONCURRENCY="${FAST_COV_SCALAR_ARRAY_CONCURRENCY:-1}"
FAST_COV_SPIN2_ARRAY_CONCURRENCY="${FAST_COV_SPIN2_ARRAY_CONCURRENCY:-4}"
FAST_COV_SERIALIZE_CLASSES="${FAST_COV_SERIALIZE_CLASSES:-0}"
FAST_COV_SCALAR_BATCH_SIZE="${FAST_COV_SCALAR_BATCH_SIZE:-10}"
FAST_COV_SPIN2_BATCH_SIZE="${FAST_COV_SPIN2_BATCH_SIZE:-8}"
FAST_COV_SCALAR_PARALLEL_GROUPS="${FAST_COV_SCALAR_PARALLEL_GROUPS:-4}"
FAST_COV_SPIN2_PARALLEL_GROUPS="${FAST_COV_SPIN2_PARALLEL_GROUPS:-8}"
FAST_COV_SCALAR_OMP_THREADS="${FAST_COV_SCALAR_OMP_THREADS:-32}"
FAST_COV_SPIN2_OMP_THREADS="${FAST_COV_SPIN2_OMP_THREADS:-16}"
MIDRES_COV_SCALAR_ARRAY_CONCURRENCY="${MIDRES_COV_SCALAR_ARRAY_CONCURRENCY:-1}"
MIDRES_COV_SPIN2_ARRAY_CONCURRENCY="${MIDRES_COV_SPIN2_ARRAY_CONCURRENCY:-1}"
MIDRES_COV_SERIALIZE_CLASSES="${MIDRES_COV_SERIALIZE_CLASSES:-0}"
MIDRES_COV_SCALAR_BATCH_SIZE="${MIDRES_COV_SCALAR_BATCH_SIZE:-5}"
MIDRES_COV_SPIN2_BATCH_SIZE="${MIDRES_COV_SPIN2_BATCH_SIZE:-4}"
MIDRES_COV_SCALAR_PARALLEL_GROUPS="${MIDRES_COV_SCALAR_PARALLEL_GROUPS:-1}"
MIDRES_COV_SPIN2_PARALLEL_GROUPS="${MIDRES_COV_SPIN2_PARALLEL_GROUPS:-1}"
MIDRES_COV_SCALAR_OMP_THREADS="${MIDRES_COV_SCALAR_OMP_THREADS:-128}"
MIDRES_COV_SPIN2_OMP_THREADS="${MIDRES_COV_SPIN2_OMP_THREADS:-128}"

usage() {
  cat <<'EOF'
Usage:
  submit_multiprobe_cpu.sh [--stages fast1024|midres2048|fast1024,midres2048] [--output-dir DIR] [--gate-midres-on-fast] [--force]

Stages:
  fast1024    nside=1024, lmax=1024, 10 linear bins
  midres2048  nside=2048, lmax=4096, 10 linear bins

Default:
  fast1024 only

Fast1024 covariance fan-out can be tuned with environment variables:
  FAST_COV_SCALAR_ARRAY_CONCURRENCY, FAST_COV_SPIN2_ARRAY_CONCURRENCY,
  FAST_COV_SERIALIZE_CLASSES, FAST_COV_*_BATCH_SIZE,
  FAST_COV_*_PARALLEL_GROUPS, FAST_COV_*_OMP_THREADS.

Midres2048 covariance fan-out uses the analogous variables:
  MIDRES_COV_SCALAR_ARRAY_CONCURRENCY, MIDRES_COV_SPIN2_ARRAY_CONCURRENCY,
  MIDRES_COV_SERIALIZE_CLASSES, MIDRES_COV_*_BATCH_SIZE,
  MIDRES_COV_*_PARALLEL_GROUPS, MIDRES_COV_*_OMP_THREADS.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stages)
      STAGES="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --gate-midres-on-fast)
      GATE_MIDRES_ON_FAST=1
      shift
      ;;
    --force)
      FORCE_FLAG="--force"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

stage_csv_contains() {
  local needle="$1"
  [[ ",${STAGES}," == *",${needle},"* ]]
}

stage_resources() {
  local stage="$1"
  local phase="$2"
  case "${stage}:${phase}" in
    fast1024:prepare|fast1024:spectra)
      echo "16 128G 10:00:00"
      ;;
    fast1024:cov-scalar)
      echo "128 128G 04:00:00"
      ;;
    fast1024:cov-spin2)
      echo "128 128G 04:00:00"
      ;;
    fast1024:assemble|fast1024:validate)
      echo "4 32G 02:00:00"
      ;;
    midres2048:prepare|midres2048:spectra)
      echo "32 256G 24:00:00"
      ;;
    midres2048:cov-scalar)
      echo "128 256G 24:00:00"
      ;;
    midres2048:cov-spin2)
      echo "128 256G 36:00:00"
      ;;
    midres2048:assemble|midres2048:validate)
      echo "8 64G 06:00:00"
      ;;
    *)
      echo "8 64G 12:00:00"
      ;;
  esac
}

sbatch_phase() {
  local stage="$1"
  local phase="$2"
  local dependency="$3"
  shift 3
  local cpus mem time
  read -r cpus mem time < <(stage_resources "${stage}" "${phase}")
  local dep_args=()
  if [[ -n "${dependency}" ]]; then
    dep_args=(--dependency="${dependency}")
  fi
  sbatch --parsable \
    --job-name="xdesi_${stage}_${phase}" \
    --cpus-per-task="${cpus}" \
    --mem="${mem}" \
    --time="${time}" \
    "${dep_args[@]}" \
    "${WORKER}" "$@"
}

manifest_file_for_stage() {
  local stage="$1"
  case "${stage}" in
    fast1024)
      echo "${REPO_ROOT}/${OUTPUT_DIR}/fast1024/covariance_manifest_nside1024_lmax1024_nbin10_linear.json"
      ;;
    midres2048)
      echo "${REPO_ROOT}/${OUTPUT_DIR}/midres2048/covariance_manifest_nside2048_lmax4096_nbin10_linear.json"
      ;;
    *)
      echo "Unsupported stage ${stage}" >&2
      exit 2
      ;;
  esac
}

manifest_count() {
  local manifest="$1"
  local cov_class="$2"
  "${PYTHON}" -c "import json,sys; m=json.load(open(sys.argv[1])); print(sum(1 for g in m['groups'] if g['class']==sys.argv[2]))" "${manifest}" "${cov_class}"
}

batch_size_for_stage_class() {
  local stage="$1"
  local cov_class="$2"
  case "${stage}:${cov_class}" in
    fast1024:scalar) echo "${FAST_COV_SCALAR_BATCH_SIZE}" ;;
    fast1024:spin2) echo "${FAST_COV_SPIN2_BATCH_SIZE}" ;;
    midres2048:scalar) echo "${MIDRES_COV_SCALAR_BATCH_SIZE}" ;;
    midres2048:spin2) echo "${MIDRES_COV_SPIN2_BATCH_SIZE}" ;;
    *) echo 1 ;;
  esac
}

parallel_groups_for_stage_class() {
  local stage="$1"
  local cov_class="$2"
  case "${stage}:${cov_class}" in
    fast1024:scalar) echo "${FAST_COV_SCALAR_PARALLEL_GROUPS}" ;;
    fast1024:spin2) echo "${FAST_COV_SPIN2_PARALLEL_GROUPS}" ;;
    midres2048:scalar) echo "${MIDRES_COV_SCALAR_PARALLEL_GROUPS}" ;;
    midres2048:spin2) echo "${MIDRES_COV_SPIN2_PARALLEL_GROUPS}" ;;
    *) echo 1 ;;
  esac
}

array_concurrency_for_stage_class() {
  local stage="$1"
  local cov_class="$2"
  case "${stage}:${cov_class}" in
    fast1024:scalar) echo "${FAST_COV_SCALAR_ARRAY_CONCURRENCY}" ;;
    fast1024:spin2) echo "${FAST_COV_SPIN2_ARRAY_CONCURRENCY}" ;;
    midres2048:scalar) echo "${MIDRES_COV_SCALAR_ARRAY_CONCURRENCY}" ;;
    midres2048:spin2) echo "${MIDRES_COV_SPIN2_ARRAY_CONCURRENCY}" ;;
    *) echo 1 ;;
  esac
}

serialize_cov_classes_for_stage() {
  local stage="$1"
  case "${stage}" in
    fast1024) echo "${FAST_COV_SERIALIZE_CLASSES}" ;;
    midres2048) echo "${MIDRES_COV_SERIALIZE_CLASSES}" ;;
    *) echo 0 ;;
  esac
}

omp_threads_for_stage_class() {
  local stage="$1"
  local cov_class="$2"
  case "${stage}:${cov_class}" in
    fast1024:scalar) echo "${FAST_COV_SCALAR_OMP_THREADS}" ;;
    fast1024:spin2) echo "${FAST_COV_SPIN2_OMP_THREADS}" ;;
    midres2048:scalar) echo "${MIDRES_COV_SCALAR_OMP_THREADS}" ;;
    midres2048:spin2) echo "${MIDRES_COV_SPIN2_OMP_THREADS}" ;;
    *) echo 1 ;;
  esac
}

ceil_div() {
  local n="$1"
  local d="$2"
  echo $(( (n + d - 1) / d ))
}

submit_stage() {
  local stage="$1"
  local stage_dependency="${2:-}"
  local common=(--stage "${stage}" --output-dir "${OUTPUT_DIR}" ${FORCE_FLAG})
  local manifest
  manifest="$(manifest_file_for_stage "${stage}")"

  echo "[submit] building covariance manifest locally for ${stage}: ${manifest}" >&2
  "${PYTHON}" "${DRIVER}" make-cov-manifest "${common[@]}"

  local prepare_dep=""
  if [[ -n "${stage_dependency}" ]]; then
    prepare_dep="afterok:${stage_dependency}"
  fi

  local prepare_job spectra_job scalar_job spin2_job assemble_job validate_job
  prepare_job="$(sbatch_phase "${stage}" prepare "${prepare_dep}" prepare "${common[@]}")"
  spectra_job="$(sbatch_phase "${stage}" spectra "afterok:${prepare_job}" spectra "${common[@]}")"

  local scalar_count spin2_count
  scalar_count="$(manifest_count "${manifest}" scalar)"
  spin2_count="$(manifest_count "${manifest}" spin2)"

  local cov_dependencies="afterok:${spectra_job}"
  local serialize_cov_classes
  serialize_cov_classes="$(serialize_cov_classes_for_stage "${stage}")"
  echo "[submit] ${stage}: covariance scalar_groups=${scalar_count} spin2_groups=${spin2_count} serialize_classes=${serialize_cov_classes}" >&2
  if [[ "${scalar_count}" -gt 0 ]]; then
    local cpus mem time
    local scalar_batch_size scalar_batch_count scalar_parallel_groups scalar_omp_threads scalar_array_concurrency
    scalar_batch_size="$(batch_size_for_stage_class "${stage}" scalar)"
    scalar_parallel_groups="$(parallel_groups_for_stage_class "${stage}" scalar)"
    scalar_omp_threads="$(omp_threads_for_stage_class "${stage}" scalar)"
    scalar_array_concurrency="$(array_concurrency_for_stage_class "${stage}" scalar)"
    scalar_batch_count="$(ceil_div "${scalar_count}" "${scalar_batch_size}")"
    read -r cpus mem time < <(stage_resources "${stage}" cov-scalar)
    echo "[submit] ${stage}: scalar batches=${scalar_batch_count} batch_size=${scalar_batch_size} array_concurrency=${scalar_array_concurrency} parallel_groups=${scalar_parallel_groups} omp_threads=${scalar_omp_threads}" >&2
    scalar_job="$(sbatch --parsable \
      --job-name="xdesi_${stage}_cov_scalar" \
      --cpus-per-task="${cpus}" \
      --mem="${mem}" \
      --time="${time}" \
      --array="0-$((scalar_batch_count - 1))%${scalar_array_concurrency}" \
      --dependency="${cov_dependencies}" \
      "${WORKER}" cov-batch "${common[@]}" --cov-class scalar --batch-size "${scalar_batch_size}" \
      --parallel-groups "${scalar_parallel_groups}" --omp-threads-per-group "${scalar_omp_threads}")"
  fi
  if [[ "${spin2_count}" -gt 0 ]]; then
    local cpus mem time
    local spin2_batch_size spin2_batch_count spin2_parallel_groups spin2_omp_threads spin2_array_concurrency spin2_dependency
    spin2_batch_size="$(batch_size_for_stage_class "${stage}" spin2)"
    spin2_parallel_groups="$(parallel_groups_for_stage_class "${stage}" spin2)"
    spin2_omp_threads="$(omp_threads_for_stage_class "${stage}" spin2)"
    spin2_array_concurrency="$(array_concurrency_for_stage_class "${stage}" spin2)"
    spin2_batch_count="$(ceil_div "${spin2_count}" "${spin2_batch_size}")"
    read -r cpus mem time < <(stage_resources "${stage}" cov-spin2)
    spin2_dependency="${cov_dependencies}"
    if [[ "${serialize_cov_classes}" -eq 1 && -n "${scalar_job:-}" ]]; then
      spin2_dependency="afterok:${scalar_job}"
    fi
    echo "[submit] ${stage}: spin2 batches=${spin2_batch_count} batch_size=${spin2_batch_size} array_concurrency=${spin2_array_concurrency} parallel_groups=${spin2_parallel_groups} omp_threads=${spin2_omp_threads} dependency=${spin2_dependency}" >&2
    spin2_job="$(sbatch --parsable \
      --job-name="xdesi_${stage}_cov_spin2" \
      --cpus-per-task="${cpus}" \
      --mem="${mem}" \
      --time="${time}" \
      --array="0-$((spin2_batch_count - 1))%${spin2_array_concurrency}" \
      --dependency="${spin2_dependency}" \
      "${WORKER}" cov-batch "${common[@]}" --cov-class spin2 --batch-size "${spin2_batch_size}" \
      --parallel-groups "${spin2_parallel_groups}" --omp-threads-per-group "${spin2_omp_threads}")"
  fi

  local assemble_dep="afterok:${spectra_job}"
  if [[ -n "${scalar_job:-}" ]]; then
    assemble_dep="${assemble_dep}:${scalar_job}"
  fi
  if [[ -n "${spin2_job:-}" ]]; then
    assemble_dep="${assemble_dep}:${spin2_job}"
  fi
  assemble_job="$(sbatch_phase "${stage}" assemble "${assemble_dep}" assemble "${common[@]}")"
  validate_job="$(sbatch_phase "${stage}" validate "afterok:${assemble_job}" validate "${common[@]}")"

  echo "[submit] ${stage}: prepare=${prepare_job} spectra=${spectra_job} scalar=${scalar_job:-none} spin2=${spin2_job:-none} assemble=${assemble_job} validate=${validate_job}" >&2
  echo "${validate_job}"
}

fast_validate_job=""
if stage_csv_contains fast1024; then
  fast_validate_job="$(submit_stage fast1024)"
fi

if stage_csv_contains midres2048; then
  dep=""
  if [[ "${GATE_MIDRES_ON_FAST}" -eq 1 && -n "${fast_validate_job}" ]]; then
    dep="${fast_validate_job}"
  fi
  submit_stage midres2048 "${dep}" >/dev/null
fi
