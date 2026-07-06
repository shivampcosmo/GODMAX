#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste

# The pz3 halo catalog already exists from preprocess. Set DO_PREPROCESS=1 only
# if the catalog is missing or the redshift/mass/cap cuts changed.
export NSIDE=${NSIDE:-512}
export NUM_SPLITS=${NUM_SPLITS:-4}
export DO_PREPROCESS=${DO_PREPROCESS:-0}
export OVERWRITE_CATALOG=${OVERWRITE_CATALOG:-0}
export MONITOR_INTERVAL=${MONITOR_INTERVAL:-180}
export GPU_SAMPLE_INTERVAL=${GPU_SAMPLE_INTERVAL:-10}

exec "${SCRIPT_DIR}/submit_stage31_pz3_cap600_pipeline.sh"
