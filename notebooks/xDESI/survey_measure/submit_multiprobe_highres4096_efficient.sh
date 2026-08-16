#!/bin/bash
set -euo pipefail

REPO_ROOT="/mnt/ceph/users/spandey/ltu-godmax/GODMAX"
PYTHON="/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python"
DRIVER="${REPO_ROOT}/notebooks/xDESI/survey_measure/run_multiprobe_production.py"
ESTIMATOR="${REPO_ROOT}/notebooks/xDESI/survey_measure/multiprobe_namaster.py"
THEORY_UTILS="${REPO_ROOT}/notebooks/xDESI/survey_measure/godmax_multiprobe_theory_utils.py"
COV_WORKER="${REPO_ROOT}/notebooks/xDESI/survey_measure/run_multiprobe_cov_bundle_worker.sbatch"
FINALIZE_WORKER="${REPO_ROOT}/notebooks/xDESI/survey_measure/run_multiprobe_finalize_worker.sbatch"
SUBMIT_SCRIPT="${REPO_ROOT}/notebooks/xDESI/survey_measure/submit_multiprobe_highres4096_efficient.sh"
LOG_DIR="${REPO_ROOT}/notebooks/xDESI/survey_measure/logs"

OUTPUT_DIR="${OUTPUT_DIR:-data/xDESI/processed/multiprobe_namaster_highres4096_ell8192_dr9random8}"
MAX_NODES="${MAX_NODES:-5}"
GROUPS_PER_NODE=11
CPUS_PER_GROUP=11
MEMORY_PER_GROUP=80G
COV_NODE_MEMORY=880G
COV_WALLTIME=04:00:00
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: submit_multiprobe_highres4096_efficient.sh [options]

Options:
  --output-dir DIR   Existing highres4096 pilot/output root.
  --max-nodes N     Covariance-node concurrency, 1..5 (default 5).
  --dry-run         Build and validate the work plan, but do not call sbatch.
  -h, --help        Show this help.

This is a resume-only production submission. It never queues map preparation or
spectra measurement; both existing products must pass identity checks first.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --max-nodes)
      MAX_NODES="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
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

if ! [[ "${MAX_NODES}" =~ ^[1-5]$ ]]; then
  echo "--max-nodes must be an integer from 1 through 5; refusing a larger submission." >&2
  exit 2
fi

if [[ "${OUTPUT_DIR}" = /* ]]; then
  OUTPUT_ROOT="${OUTPUT_DIR}"
else
  OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_DIR}"
fi
STAGE_ROOT="${OUTPUT_ROOT}/highres4096"
TAG="nside4096_ell128_lmax8192_lmask12287_nbin20_log_pipev2"
PRODUCT_TAG="${TAG}_gshot_gkell3000_dvvalidv1"
MANIFEST_PATH="${STAGE_ROOT}/covariance_manifest_${TAG}.json"
PLAN_PATH="${STAGE_ROOT}/covariance_work_plan_${TAG}.json"
MAPS_PATH="${STAGE_ROOT}/xdesi_multiprobe_maps_${TAG}.h5"
SPECTRA_PATH="${STAGE_ROOT}/xdesi_multiprobe_spectra_${PRODUCT_TAG}.h5"

mkdir -p "${LOG_DIR}" "${STAGE_ROOT}"
cd "${REPO_ROOT}"

# Serialize preflight/submission and refuse every second invocation before it
# can rewrite the plan file frozen into an active DAG (including --dry-run).
exec 9>"${STAGE_ROOT}/.submit_highres4096_efficient.lock"
if ! flock -n 9; then
  echo "Another efficient highres4096 preflight/submission holds the stage lock." >&2
  exit 4
fi
active_same_name="$(squeue -u "${USER}" -h \
  -n xdesi_highres4096_cov_stress,xdesi_highres4096_cov_main,xdesi_highres4096_finalize \
  -o '%A' | head -n 1)"
if [[ -n "${active_same_name}" ]]; then
  echo "An efficient highres4096 production job is already active (${active_same_name}); refusing to rewrite its frozen work plan." >&2
  exit 4
fi

RUNTIME_SOURCE_FILES=(
  "${DRIVER}"
  "${ESTIMATOR}"
  "${THEORY_UTILS}"
  "${COV_WORKER}"
  "${FINALIZE_WORKER}"
  "${SUBMIT_SCRIPT}"
)
export XDESI_RUNTIME_SOURCE_FILES
XDESI_RUNTIME_SOURCE_FILES="$(IFS=:; echo "${RUNTIME_SOURCE_FILES[*]}")"
export XDESI_RUNTIME_SOURCE_SHA256
XDESI_RUNTIME_SOURCE_SHA256="$(sha256sum "${RUNTIME_SOURCE_FILES[@]}" | sha256sum | cut -d' ' -f1)"

echo "[preflight] runtime_source_sha256=${XDESI_RUNTIME_SOURCE_SHA256}" >&2
echo "[preflight] reusing map=${MAPS_PATH}" >&2
echo "[preflight] reusing spectra=${SPECTRA_PATH}" >&2

"${PYTHON}" "${DRIVER}" make-cov-manifest \
  --stage highres4096 \
  --output-dir "${OUTPUT_DIR}" \
  --manifest-out "${MANIFEST_PATH}"
"${PYTHON}" "${DRIVER}" make-cov-work-plan \
  --stage highres4096 \
  --output-dir "${OUTPUT_DIR}" \
  --maps-path "${MAPS_PATH}" \
  --spectra-path "${SPECTRA_PATH}" \
  --manifest-path "${MANIFEST_PATH}" \
  --plan-out "${PLAN_PATH}" \
  --groups-per-bundle "${GROUPS_PER_NODE}"

export XDESI_COV_WORK_PLAN_SHA256
XDESI_COV_WORK_PLAN_SHA256="$(sha256sum "${PLAN_PATH}" | cut -d' ' -f1)"
read -r n_reused n_missing n_bundles stress_groups < <(
  "${PYTHON}" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); print(p["n_reused_groups"],p["n_missing_groups"],p["n_bundles"],",".join(map(str,p["stress_group_indices"])))' \
    "${PLAN_PATH}"
)
echo "[preflight] plan_sha256=${XDESI_COV_WORK_PLAN_SHA256} reused=${n_reused} missing=${n_missing} bundles=${n_bundles}" >&2
echo "[preflight] production stress groups=${stress_groups}" >&2

if [[ "${n_missing}" -gt 0 && "${n_bundles}" -lt 1 ]]; then
  echo "Work plan reports missing groups but no bundles." >&2
  exit 3
fi
if [[ "${n_bundles}" -gt 0 ]]; then
  largest_bundle="$(${PYTHON} -c 'import json,sys; p=json.load(open(sys.argv[1])); print(max(len(x["group_indices"]) for x in p["bundles"]))' "${PLAN_PATH}")"
  if [[ "${largest_bundle}" -gt "${GROUPS_PER_NODE}" ]]; then
    echo "Work plan exceeds ${GROUPS_PER_NODE} groups per node." >&2
    exit 3
  fi
fi

normalize_job_id() {
  local raw="$1"
  echo "${raw%%;*}"
}

submit_or_echo() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] sbatch $*" >&2
    local arg label="job"
    for arg in "$@"; do
      if [[ "${arg}" == --job-name=* ]]; then
        label="${arg#--job-name=}"
        break
      fi
    done
    echo "dryrun-${label}"
  else
    sbatch --parsable "$@"
  fi
}

stress_job=""
main_job=""
final_dependency=""
if [[ "${n_bundles}" -gt 0 ]]; then
  stress_raw="$(submit_or_echo \
    --job-name=xdesi_highres4096_cov_stress \
    --nodes=1 --ntasks=11 --cpus-per-task=11 \
    --constraint=rome --partition=cmbas --mem="${COV_NODE_MEMORY}" --time="${COV_WALLTIME}" \
    --export=ALL,XDESI_COV_BUNDLE_ID=0 \
    "${COV_WORKER}" "${PLAN_PATH}" "${OUTPUT_DIR}" "${MANIFEST_PATH}" "${MAPS_PATH}")"
  stress_job="$(normalize_job_id "${stress_raw}")"
  final_dependency="afterok:${stress_job}"

  if [[ "${n_bundles}" -gt 1 ]]; then
    main_raw="$(submit_or_echo \
      --job-name=xdesi_highres4096_cov_main \
      --nodes=1 --ntasks=11 --cpus-per-task=11 \
      --constraint=rome --partition=cmbas --mem="${COV_NODE_MEMORY}" --time="${COV_WALLTIME}" \
      --array="1-$((n_bundles - 1))%${MAX_NODES}" \
      --dependency="afterok:${stress_job}" \
      "${COV_WORKER}" "${PLAN_PATH}" "${OUTPUT_DIR}" "${MANIFEST_PATH}" "${MAPS_PATH}")"
    main_job="$(normalize_job_id "${main_raw}")"
    final_dependency="afterok:${main_job}"
  fi
fi

final_args=(
  --job-name=xdesi_highres4096_finalize
  --nodes=1 --ntasks=1 --cpus-per-task=2
  --partition=genx --mem=4G --time=00:30:00
)
if [[ -n "${final_dependency}" ]]; then
  final_args+=(--dependency="${final_dependency}")
fi
final_raw="$(submit_or_echo "${final_args[@]}" \
  "${FINALIZE_WORKER}" "${PLAN_PATH}" "${OUTPUT_DIR}" "${MANIFEST_PATH}")"
final_job="$(normalize_job_id "${final_raw}")"

record_path="${STAGE_ROOT}/submission_highres4096_efficient_${XDESI_COV_WORK_PLAN_SHA256:0:12}.json"
"${PYTHON}" -c \
  'import json,sys; p={"runtime_source_sha256":sys.argv[2],"work_plan_path":sys.argv[3],"work_plan_file_sha256":sys.argv[4],"max_covariance_nodes":int(sys.argv[5]),"groups_per_node":11,"cpus_per_group":11,"memory_per_group":"80G","covariance_node_memory":"880G","covariance_walltime":"04:00:00","stress_job":sys.argv[6],"main_array_job":sys.argv[7],"finalize_job":sys.argv[8],"dry_run":bool(int(sys.argv[9])),"reused_groups":int(sys.argv[10]),"missing_groups":int(sys.argv[11]),"bundle_count":int(sys.argv[12]),"map_and_spectra_recomputed":False}; open(sys.argv[1],"w").write(json.dumps(p,indent=2,sort_keys=True)+"\n")' \
  "${record_path}" "${XDESI_RUNTIME_SOURCE_SHA256}" "${PLAN_PATH}" "${XDESI_COV_WORK_PLAN_SHA256}" \
  "${MAX_NODES}" "${stress_job}" "${main_job}" "${final_job}" "${DRY_RUN}" \
  "${n_reused}" "${n_missing}" "${n_bundles}"

echo "[submit] stress=${stress_job:-none} main=${main_job:-none} finalize=${final_job}" >&2
echo "[submit] hard covariance-node cap=${MAX_NODES}; prepare/spectra jobs submitted=0" >&2
echo "[submit] record=${record_path}" >&2
