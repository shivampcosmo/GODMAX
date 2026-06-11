#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste
NSIDE=${NSIDE:-1024}
NUM_SPLITS=${NUM_SPLITS:-4}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-180}
GPU_SAMPLE_INTERVAL=${GPU_SAMPLE_INTERVAL:-10}
DO_PREPROCESS=${DO_PREPROCESS:-1}
OVERWRITE_CATALOG=${OVERWRITE_CATALOG:-0}

mkdir -p "${SCRIPT_DIR}/slurm_logs"

dependency=()
if [ "${DO_PREPROCESS}" -gt 0 ]; then
  preprocess_job=$(sbatch --parsable \
    --export=ALL,MONITOR_INTERVAL="${MONITOR_INTERVAL}",GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL}",OVERWRITE_CATALOG="${OVERWRITE_CATALOG}" \
    "${SCRIPT_DIR}/submit_stage31_pz3_cap600_preprocess.sbatch")
  dependency=(--dependency=afterok:"${preprocess_job}")
  echo "Submitted preprocess job: ${preprocess_job}"
else
  echo "Skipping preprocess; assuming the pz3 cap halo catalog already exists."
fi

paste_job=$(sbatch --parsable \
  "${dependency[@]}" \
  --array=0-$((NUM_SPLITS - 1))%"${NUM_SPLITS}" \
  --export=ALL,NSIDE="${NSIDE}",NUM_SPLITS="${NUM_SPLITS}",MONITOR_INTERVAL="${MONITOR_INTERVAL}",GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL}" \
  "${SCRIPT_DIR}/submit_stage31_pz3_cap600_paste_array.sbatch")

combine_job=$(sbatch --parsable \
  --dependency=afterok:"${paste_job}" \
  --export=ALL,NSIDE="${NSIDE}",NUM_SPLITS="${NUM_SPLITS}",MONITOR_INTERVAL="${MONITOR_INTERVAL}",GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL}" \
  "${SCRIPT_DIR}/submit_stage31_pz3_cap600_combine.sbatch")

echo "Submitted paste array job: ${paste_job}"
echo "Submitted dependent combine/measure/plot job: ${combine_job}"
echo "Monitor interval: ${MONITOR_INTERVAL}s; GPU sample interval: ${GPU_SAMPLE_INTERVAL}s"
echo "Logs: ${SCRIPT_DIR}/slurm_logs"
