#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Dedicated true-n(z) midres run:
#   stage=midres2048, nside=2048, lmax=4096, nbin=10, linear bins.
#
# This uses the corrected photometric-catalog handling in multiprobe_namaster.py:
# the full valid_for_cl DR9 Extended LRG catalog is used for maps/spectra, while
# calibrated true-redshift DESI kernels are written to nz/desi/* for theory.
DEFAULT_STAGES="${STAGES:-midres2048}"
DEFAULT_OUTPUT_DIR="${OUTPUT_DIR:-data/xDESI/processed/multiprobe_namaster_true_nz}"

# Keep the large covariance fan-out bounded. Scalar groups are few, so allow two
# nodes there; spin2 groups dominate and may use up to eight full CPU nodes.
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
