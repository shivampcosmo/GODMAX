#!/bin/bash
#SBATCH --job-name=msbi_found
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/foundations_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/foundations_%j.err
#
# Gate 1 (measurement chain) and the round-1 design guide.  Both already PASS
# interactively; this script exists so they are reproducible from a clean job.
#
#   step 1  the archived paste, measured through this module, must reproduce the noise
#           contract's stored bandpowers, and mu_paste + nu(observation seeds) must
#           reproduce the inference contract's data_vector.  If this fails, no
#           mock-SBI vector is comparable to what the theory runs used.
#   step 2  the one-point transfer r_hat and the Laplace guide, with a hard gate that
#           the guide brackets the frozen paste's own parameter point.  The RAW theory
#           guide puts it at the 99.999th percentile; the transfer-corrected guide must
#           do far better or the design would be spent where the paste model cannot fit.
#   step 3  backend/forward non-regression against the pinned parity vectors.

set -euo pipefail
REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
OUT=$REPO/data/SBI_validate/mock_sbi
export JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=/tmp/matplotlib
cd "$REPO/notebooks/SBI_validate"

echo "### step 1: measurement-chain foundations"
$PY validate_mock_sbi_foundations.py --output "$OUT/foundations.json"

echo "### step 2: transfer and round-1 guide"
$PY build_mock_sbi_transfer.py --output-dir "$OUT"

echo "### step 3: forward-model non-regression (parity vectors must be unchanged)"
$PY check_three_probe_backend_parity.py \
    --reference-point "$REPO/data/SBI_validate/three_probe_inference/reference_point_v2.json" \
    --contract "$REPO/data/SBI_validate/three_probe_inference/inference_contract.yaml" \
    --output "$OUT/backend_parity.json"
