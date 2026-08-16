#!/bin/bash
set -euo pipefail

REPO_ROOT="/mnt/ceph/users/spandey/ltu-godmax/GODMAX"
PYTHON="/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python"
DRIVER="${REPO_ROOT}/notebooks/xDESI/survey_measure/run_multiprobe_production.py"
WORKER="${REPO_ROOT}/notebooks/xDESI/survey_measure/run_multiprobe_cpu_worker.sbatch"
LOG_DIR="${REPO_ROOT}/notebooks/xDESI/survey_measure/logs"
OUTPUT_DIR="${OUTPUT_DIR:-data/xDESI/processed/multiprobe_namaster}"
STAGES="${STAGES:-fast1024}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-xdesi-${USER}}"
GATE_MIDRES_ON_FAST="${GATE_MIDRES_ON_FAST:-0}"
# Legacy compatibility variable. Pipeline v2 changes masks and bandpower windows for all
# spectra with a shear endpoint, so partial in-place patching is deliberately forbidden.
PATCH_SHEAR_SPECTRA="${PATCH_SHEAR_SPECTRA:-0}"
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
MIDRES_COV_SCALAR_ARRAY_CONCURRENCY="${MIDRES_COV_SCALAR_ARRAY_CONCURRENCY:-10}"
MIDRES_COV_SPIN2_ARRAY_CONCURRENCY="${MIDRES_COV_SPIN2_ARRAY_CONCURRENCY:-96}"
MIDRES_COV_SERIALIZE_CLASSES="${MIDRES_COV_SERIALIZE_CLASSES:-0}"
MIDRES_COV_SCALAR_BATCH_SIZE="${MIDRES_COV_SCALAR_BATCH_SIZE:-1}"
MIDRES_COV_SPIN2_BATCH_SIZE="${MIDRES_COV_SPIN2_BATCH_SIZE:-1}"
MIDRES_COV_SCALAR_PARALLEL_GROUPS="${MIDRES_COV_SCALAR_PARALLEL_GROUPS:-1}"
MIDRES_COV_SPIN2_PARALLEL_GROUPS="${MIDRES_COV_SPIN2_PARALLEL_GROUPS:-1}"
MIDRES_COV_SCALAR_OMP_THREADS="${MIDRES_COV_SCALAR_OMP_THREADS:-128}"
MIDRES_COV_SPIN2_OMP_THREADS="${MIDRES_COV_SPIN2_OMP_THREADS:-128}"
HIGHRES_COV_SCALAR_ARRAY_CONCURRENCY="${HIGHRES_COV_SCALAR_ARRAY_CONCURRENCY:-10}"
HIGHRES_COV_SPIN2_ARRAY_CONCURRENCY="${HIGHRES_COV_SPIN2_ARRAY_CONCURRENCY:-29}"
HIGHRES_COV_SERIALIZE_CLASSES="${HIGHRES_COV_SERIALIZE_CLASSES:-1}"
HIGHRES_COV_SCALAR_BATCH_SIZE="${HIGHRES_COV_SCALAR_BATCH_SIZE:-1}"
HIGHRES_COV_SPIN2_BATCH_SIZE="${HIGHRES_COV_SPIN2_BATCH_SIZE:-1}"
HIGHRES_COV_SCALAR_PARALLEL_GROUPS="${HIGHRES_COV_SCALAR_PARALLEL_GROUPS:-1}"
HIGHRES_COV_SPIN2_PARALLEL_GROUPS="${HIGHRES_COV_SPIN2_PARALLEL_GROUPS:-1}"
HIGHRES_COV_SCALAR_OMP_THREADS="${HIGHRES_COV_SCALAR_OMP_THREADS:-128}"
HIGHRES_COV_SPIN2_OMP_THREADS="${HIGHRES_COV_SPIN2_OMP_THREADS:-128}"
PLOT_ELL_MAX="${PLOT_ELL_MAX:-0}"
PLOT_KSZ_YLIM="${PLOT_KSZ_YLIM:-auto}"

# Every queued phase executes the shared worktree later. Bind the whole DAG to
# the exact runtime source bytes present at submission so a subsequent edit
# fails closed instead of silently mixing estimator versions across shards.
RUNTIME_SOURCE_FILES=(
  "${DRIVER}"
  "${REPO_ROOT}/notebooks/xDESI/survey_measure/multiprobe_namaster.py"
  "${REPO_ROOT}/notebooks/xDESI/survey_measure/godmax_multiprobe_theory_utils.py"
  "${WORKER}"
)
runtime_source_digest() {
  sha256sum "${RUNTIME_SOURCE_FILES[@]}" | sha256sum | cut -d' ' -f1
}
export XDESI_RUNTIME_SOURCE_FILES
XDESI_RUNTIME_SOURCE_FILES="$(IFS=:; echo "${RUNTIME_SOURCE_FILES[*]}")"
export XDESI_RUNTIME_SOURCE_SHA256
XDESI_RUNTIME_SOURCE_SHA256="$(runtime_source_digest)"

usage() {
  cat <<'EOF'
Usage:
  submit_multiprobe_cpu.sh [--stages fast1024|midres2048|highres4096|CSV] [--output-dir DIR] [--gate-midres-on-fast] [--force]

Stages:
  fast1024    nside=1024, lmax=1024, 10 linear bins
  midres2048  nside=2048, ell=128..4096, 16 hybrid-log bins; lmax_mask=6143
  highres4096 nside=4096, ell=128..8192, 20 hybrid-log bins; lmax_mask=12287;
              ACT-kappa bands above ell=3000 are archived as invalid zero placeholders

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

Highres4096 uses HIGHRES_COV_* variables, one group per task, serial covariance
classes, and disables the ~4-GiB/group on-disk covariance-workspace cache.
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

IFS=',' read -r -a requested_stages <<< "${STAGES}"
for requested_stage in "${requested_stages[@]}"; do
  case "${requested_stage}" in
    fast1024|midres2048|highres4096) ;;
    *)
      echo "Unsupported stage '${requested_stage}'; expected fast1024, midres2048, or highres4096." >&2
      exit 2
      ;;
  esac
done

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"
echo "[submit] runtime_source_sha256=${XDESI_RUNTIME_SOURCE_SHA256}" >&2

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
    fast1024:assemble|fast1024:validate|fast1024:plot-dell|fast1024:plot-cl-dell)
      echo "4 32G 02:00:00"
      ;;
    midres2048:prepare|midres2048:spectra)
      # 128 cores select a whole Rome node; measured peak RSS stays below 64 GiB.
      echo "128 128G 02:00:00"
      ;;
    midres2048:cov-scalar)
      echo "128 128G 04:00:00"
      ;;
    midres2048:cov-spin2)
      echo "128 128G 04:00:00"
      ;;
    midres2048:assemble|midres2048:validate|midres2048:plot-dell|midres2048:plot-cl-dell)
      echo "8 64G 01:00:00"
      ;;
    highres4096:prepare)
      echo "128 512G 04:00:00"
      ;;
    highres4096:spectra)
      echo "128 768G 12:00:00"
      ;;
    highres4096:cov-scalar|highres4096:cov-spin2)
      echo "128 512G 12:00:00"
      ;;
    highres4096:assemble|highres4096:validate|highres4096:plot-dell|highres4096:plot-cl-dell)
      echo "8 64G 02:00:00"
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
  local output_root
  if [[ "${OUTPUT_DIR}" = /* ]]; then
    output_root="${OUTPUT_DIR}"
  else
    output_root="${REPO_ROOT}/${OUTPUT_DIR}"
  fi
  case "${stage}" in
    fast1024)
      echo "${output_root}/fast1024/covariance_manifest_nside1024_lmax1024_nbin10_linear_pipev2.json"
      ;;
    midres2048)
      echo "${output_root}/midres2048/covariance_manifest_nside2048_ell128_lmax4096_lmask6143_nbin16_log_pipev2.json"
      ;;
    highres4096)
      echo "${output_root}/highres4096/covariance_manifest_nside4096_ell128_lmax8192_lmask12287_nbin20_log_pipev2.json"
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
    highres4096:scalar) echo "${HIGHRES_COV_SCALAR_BATCH_SIZE}" ;;
    highres4096:spin2) echo "${HIGHRES_COV_SPIN2_BATCH_SIZE}" ;;
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
    highres4096:scalar) echo "${HIGHRES_COV_SCALAR_PARALLEL_GROUPS}" ;;
    highres4096:spin2) echo "${HIGHRES_COV_SPIN2_PARALLEL_GROUPS}" ;;
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
    highres4096:scalar) echo "${HIGHRES_COV_SCALAR_ARRAY_CONCURRENCY}" ;;
    highres4096:spin2) echo "${HIGHRES_COV_SPIN2_ARRAY_CONCURRENCY}" ;;
    *) echo 1 ;;
  esac
}

serialize_cov_classes_for_stage() {
  local stage="$1"
  case "${stage}" in
    fast1024) echo "${FAST_COV_SERIALIZE_CLASSES}" ;;
    midres2048) echo "${MIDRES_COV_SERIALIZE_CLASSES}" ;;
    highres4096) echo "${HIGHRES_COV_SERIALIZE_CLASSES}" ;;
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
    highres4096:scalar) echo "${HIGHRES_COV_SCALAR_OMP_THREADS}" ;;
    highres4096:spin2) echo "${HIGHRES_COV_SPIN2_OMP_THREADS}" ;;
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
  local cov_cache_flag=()
  if [[ "${stage}" == "highres4096" ]]; then
    cov_cache_flag=(--no-cov-workspace-cache)
  fi
  local manifest
  manifest="$(manifest_file_for_stage "${stage}")"

  echo "[submit] building covariance manifest locally for ${stage}: ${manifest}" >&2
  "${PYTHON}" "${DRIVER}" make-cov-manifest "${common[@]}"

  local prepare_dep=""
  if [[ -n "${stage_dependency}" ]]; then
    prepare_dep="afterok:${stage_dependency}"
  fi

  local prepare_job spectra_job scalar_job spin2_job assemble_job validate_job plot_job
  prepare_job="$(sbatch_phase "${stage}" prepare "${prepare_dep}" prepare "${common[@]}")"
  local spectra_flag=""
  if [[ "${PATCH_SHEAR_SPECTRA}" -eq 1 ]]; then
    echo "[submit] PATCH_SHEAR_SPECTRA=1 is unsafe for pipeline v2; run a full spectra phase." >&2
    exit 2
  fi
  spectra_job="$(sbatch_phase "${stage}" spectra "afterok:${prepare_job}" spectra "${common[@]}" ${spectra_flag})"

  local scalar_count spin2_count
  scalar_count="$(manifest_count "${manifest}" scalar)"
  spin2_count="$(manifest_count "${manifest}" spin2)"

  # Covariance reads only the map product (never the spectra product), so it can run in
  # PARALLEL with the spectra phase -- depend on prepare, not spectra. assemble still waits
  # for spectra (below), so the data vector and covariance are joined correctly.
  local cov_dependencies="afterok:${prepare_job}"
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
      --parallel-groups "${scalar_parallel_groups}" --omp-threads-per-group "${scalar_omp_threads}" \
      "${cov_cache_flag[@]}")"
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
      --parallel-groups "${spin2_parallel_groups}" --omp-threads-per-group "${spin2_omp_threads}" \
      "${cov_cache_flag[@]}")"
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
  plot_job="$(sbatch_phase "${stage}" plot-cl-dell "afterok:${validate_job}" plot-measurement-cl-dell "${common[@]}" \
    --plot-ell-max "${PLOT_ELL_MAX}" \
    --plot-ksz-ylim="${PLOT_KSZ_YLIM}")"

  echo "[submit] ${stage}: prepare=${prepare_job} spectra=${spectra_job} scalar=${scalar_job:-none} spin2=${spin2_job:-none} assemble=${assemble_job} validate=${validate_job} plot=${plot_job}" >&2
  echo "${plot_job}"
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

if stage_csv_contains highres4096; then
  submit_stage highres4096 >/dev/null
fi
