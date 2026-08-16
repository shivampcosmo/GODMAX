#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Dedicated true-n(z) midres run:
#   stage=midres2048, nside=2048, ell=128..4096, nbin=16, hybrid-log bins,
#   lmax_mask=6143, native upstream masks, and global masked-mean subtraction.
#
# This uses the corrected photometric-catalog handling in multiprobe_namaster.py:
# the full valid_for_cl DR9 Extended LRG catalog is used for maps/spectra, while
# calibrated true-redshift DESI kernels are written to nz/desi/* for theory.
DEFAULT_STAGES="${STAGES:-midres2048}"
DEFAULT_OUTPUT_DIR="${OUTPUT_DIR:-data/xDESI/processed/multiprobe_namaster_true_nz}"

# Historical nside=2048 accounting gives a latency knee at 96 concurrent
# spin-2 nodes. Run one group per node (249 spin-2, 10 scalar groups), allow
# both arrays concurrently, and leave the scientific kernels all 128 cores.
export MIDRES_COV_SCALAR_ARRAY_CONCURRENCY="${MIDRES_COV_SCALAR_ARRAY_CONCURRENCY:-10}"
export MIDRES_COV_SPIN2_ARRAY_CONCURRENCY="${MIDRES_COV_SPIN2_ARRAY_CONCURRENCY:-96}"
export MIDRES_COV_SERIALIZE_CLASSES="${MIDRES_COV_SERIALIZE_CLASSES:-0}"
export MIDRES_COV_SCALAR_BATCH_SIZE="${MIDRES_COV_SCALAR_BATCH_SIZE:-1}"
export MIDRES_COV_SPIN2_BATCH_SIZE="${MIDRES_COV_SPIN2_BATCH_SIZE:-1}"
export MIDRES_COV_SCALAR_PARALLEL_GROUPS="${MIDRES_COV_SCALAR_PARALLEL_GROUPS:-1}"
export MIDRES_COV_SPIN2_PARALLEL_GROUPS="${MIDRES_COV_SPIN2_PARALLEL_GROUPS:-1}"
export MIDRES_COV_SCALAR_OMP_THREADS="${MIDRES_COV_SCALAR_OMP_THREADS:-128}"
export MIDRES_COV_SPIN2_OMP_THREADS="${MIDRES_COV_SPIN2_OMP_THREADS:-128}"
export PLOT_ELL_MAX="${PLOT_ELL_MAX:-0}"
export PLOT_KSZ_YLIM="${PLOT_KSZ_YLIM:-auto}"

exec "${SCRIPT_DIR}/submit_multiprobe_cpu.sh" \
  --stages "${DEFAULT_STAGES}" \
  --output-dir "${DEFAULT_OUTPUT_DIR}" \
  "$@"
