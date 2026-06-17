#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Dedicated true-n(z) midres run:
#   stage=midres2048, nside=2048, ell=128..3000, nbin=13, hybrid-log bins,
#   1 deg C2 mask apodization, and pair-overlap mean subtraction.
#
# This uses the corrected photometric-catalog handling in multiprobe_namaster.py:
# the full valid_for_cl DR9 Extended LRG catalog is used for maps/spectra, while
# calibrated true-redshift DESI kernels are written to nz/desi/* for theory.
DEFAULT_STAGES="${STAGES:-midres2048}"
DEFAULT_OUTPUT_DIR="${OUTPUT_DIR:-data/xDESI/processed/multiprobe_namaster_true_nz}"

# NaMaster covariance is SINGLE-THREADED here (measured cpu_recent=1.00 core regardless of
# OMP), so the only way to fill an exclusive 128-core/1TB rome node is to run many groups
# in parallel at OMP=1. The cov-key path now (a) loads/builds ONLY the few fields a group
# references (raw map floor 6 GB -> 1.6 GB) and (b) caches covariance workspaces to disk.
# Peak RSS drops from ~24 GB to ~14 GB/group -- the remaining floor is the transient
# gaussian_covariance working set (~9 GB), which does NOT shrink, so ~56 groups fit in
# ~990 GB. BATCH_SIZE = groups per array task; 249 spin2 groups, batch 125 x concurrency 2
# = 2 full nodes x 56 parallel. Check `seff` RSS on the first run and tune PARALLEL_GROUPS
# (48 = very safe, 64 = edge). Present shards skip on an early exit; cached workspaces are
# reused across reruns (delete <block_dir>/cov_workspaces only if you change masks/maps).
export MIDRES_COV_SCALAR_ARRAY_CONCURRENCY="${MIDRES_COV_SCALAR_ARRAY_CONCURRENCY:-1}"
export MIDRES_COV_SPIN2_ARRAY_CONCURRENCY="${MIDRES_COV_SPIN2_ARRAY_CONCURRENCY:-2}"
export MIDRES_COV_SERIALIZE_CLASSES="${MIDRES_COV_SERIALIZE_CLASSES:-1}"
export MIDRES_COV_SCALAR_BATCH_SIZE="${MIDRES_COV_SCALAR_BATCH_SIZE:-10}"
export MIDRES_COV_SPIN2_BATCH_SIZE="${MIDRES_COV_SPIN2_BATCH_SIZE:-125}"
export MIDRES_COV_SCALAR_PARALLEL_GROUPS="${MIDRES_COV_SCALAR_PARALLEL_GROUPS:-10}"
export MIDRES_COV_SPIN2_PARALLEL_GROUPS="${MIDRES_COV_SPIN2_PARALLEL_GROUPS:-56}"
export MIDRES_COV_SCALAR_OMP_THREADS="${MIDRES_COV_SCALAR_OMP_THREADS:-1}"
export MIDRES_COV_SPIN2_OMP_THREADS="${MIDRES_COV_SPIN2_OMP_THREADS:-1}"

exec "${SCRIPT_DIR}/submit_multiprobe_cpu.sh" \
  --stages "${DEFAULT_STAGES}" \
  --output-dir "${DEFAULT_OUTPUT_DIR}" \
  "$@"
