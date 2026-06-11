#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# True-n(z) production wrapper. By default this reproduces the fast1024 run,
# but --stages may be passed through to run midres2048 or both stages. The map
# writer stores calibrated true-redshift DESI n(z) under nz/desi/* and keeps the
# catalog photo-z histogram only under nz/desi_photoz_diagnostic/*.
DEFAULT_STAGES="${STAGES:-fast1024}"
DEFAULT_OUTPUT_DIR="${OUTPUT_DIR:-data/xDESI/processed/multiprobe_namaster_true_nz}"

# Fast1024 defaults: keep scalar covariance on one full node, then fan the
# expensive spin2 covariance over eight full nodes.
export FAST_COV_SCALAR_ARRAY_CONCURRENCY="${FAST_COV_SCALAR_ARRAY_CONCURRENCY:-1}"
export FAST_COV_SPIN2_ARRAY_CONCURRENCY="${FAST_COV_SPIN2_ARRAY_CONCURRENCY:-8}"
export FAST_COV_SERIALIZE_CLASSES="${FAST_COV_SERIALIZE_CLASSES:-1}"
export FAST_COV_SCALAR_BATCH_SIZE="${FAST_COV_SCALAR_BATCH_SIZE:-10}"
export FAST_COV_SPIN2_BATCH_SIZE="${FAST_COV_SPIN2_BATCH_SIZE:-8}"
export FAST_COV_SCALAR_PARALLEL_GROUPS="${FAST_COV_SCALAR_PARALLEL_GROUPS:-4}"
export FAST_COV_SPIN2_PARALLEL_GROUPS="${FAST_COV_SPIN2_PARALLEL_GROUPS:-8}"
export FAST_COV_SCALAR_OMP_THREADS="${FAST_COV_SCALAR_OMP_THREADS:-32}"
export FAST_COV_SPIN2_OMP_THREADS="${FAST_COV_SPIN2_OMP_THREADS:-16}"

# Midres2048 defaults: lmax=4096 is dominated by spin2 covariance groups. Use
# at most eight full CPU nodes for the spin2 fan-out and serialize scalar before
# spin2 so the wrapper does not accidentally request two large arrays at once.
export MIDRES_COV_SCALAR_ARRAY_CONCURRENCY="${MIDRES_COV_SCALAR_ARRAY_CONCURRENCY:-2}"
export MIDRES_COV_SPIN2_ARRAY_CONCURRENCY="${MIDRES_COV_SPIN2_ARRAY_CONCURRENCY:-8}"
export MIDRES_COV_SERIALIZE_CLASSES="${MIDRES_COV_SERIALIZE_CLASSES:-1}"
export MIDRES_COV_SCALAR_BATCH_SIZE="${MIDRES_COV_SCALAR_BATCH_SIZE:-5}"
export MIDRES_COV_SPIN2_BATCH_SIZE="${MIDRES_COV_SPIN2_BATCH_SIZE:-4}"
export MIDRES_COV_SCALAR_PARALLEL_GROUPS="${MIDRES_COV_SCALAR_PARALLEL_GROUPS:-1}"
export MIDRES_COV_SPIN2_PARALLEL_GROUPS="${MIDRES_COV_SPIN2_PARALLEL_GROUPS:-1}"
export MIDRES_COV_SCALAR_OMP_THREADS="${MIDRES_COV_SCALAR_OMP_THREADS:-128}"
export MIDRES_COV_SPIN2_OMP_THREADS="${MIDRES_COV_SPIN2_OMP_THREADS:-128}"

exec "${SCRIPT_DIR}/submit_multiprobe_cpu.sh" \
  --stages "${DEFAULT_STAGES}" \
  --output-dir "${DEFAULT_OUTPUT_DIR}" \
  "$@"
