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

# NaMaster covariance is SINGLE-THREADED *and memory-bandwidth bound* (each build/block
# churns a ~0.5 GB workspace + ~9 GB transient). MEASURED THE HARD WAY: packing 56 builds
# per node saturated the memory bus -> each process got cpu_recent~0.05 cores and a build
# that takes ~5 min solo took ~1h50m (~20x slowdown). So do NOT pack densely -- run a FEW
# groups per node and parallelize ACROSS nodes instead. PARALLEL_GROUPS=4 keeps each process
# near full speed; spread the work with ARRAY_CONCURRENCY (nodes). With ~78 groups to (re)do,
# batch 16 x concurrency 16 = 16 nodes x 4 parallel (~1 wave). VERIFY on the first heartbeat
# that cpu_recent is ~0.6-1.0 cores; if it is well below ~0.4, drop PARALLEL_GROUPS to 2.
# Present shards skip on an early exit and cached workspaces are reused, so reruns resume.
export MIDRES_COV_SCALAR_ARRAY_CONCURRENCY="${MIDRES_COV_SCALAR_ARRAY_CONCURRENCY:-1}"
export MIDRES_COV_SPIN2_ARRAY_CONCURRENCY="${MIDRES_COV_SPIN2_ARRAY_CONCURRENCY:-16}"
export MIDRES_COV_SERIALIZE_CLASSES="${MIDRES_COV_SERIALIZE_CLASSES:-1}"
export MIDRES_COV_SCALAR_BATCH_SIZE="${MIDRES_COV_SCALAR_BATCH_SIZE:-10}"
export MIDRES_COV_SPIN2_BATCH_SIZE="${MIDRES_COV_SPIN2_BATCH_SIZE:-16}"
export MIDRES_COV_SCALAR_PARALLEL_GROUPS="${MIDRES_COV_SCALAR_PARALLEL_GROUPS:-4}"
export MIDRES_COV_SPIN2_PARALLEL_GROUPS="${MIDRES_COV_SPIN2_PARALLEL_GROUPS:-4}"
export MIDRES_COV_SCALAR_OMP_THREADS="${MIDRES_COV_SCALAR_OMP_THREADS:-1}"
export MIDRES_COV_SPIN2_OMP_THREADS="${MIDRES_COV_SPIN2_OMP_THREADS:-1}"

exec "${SCRIPT_DIR}/submit_multiprobe_cpu.sh" \
  --stages "${DEFAULT_STAGES}" \
  --output-dir "${DEFAULT_OUTPUT_DIR}" \
  "$@"
