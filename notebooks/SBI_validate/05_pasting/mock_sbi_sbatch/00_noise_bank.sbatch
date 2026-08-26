#!/bin/bash
#SBATCH --job-name=msbi_noisebank
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/noise_bank_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/noise_bank_%j.err
#
# Stage 1 product: the theta-independent noise bank.
#
# Because the galaxy leg is frozen, x(theta, seed) = mu_paste(theta) + nu(seed) holds
# to machine precision with nu independent of theta (verified: 8.4e-15 relative).  So
# the noise is measured once and reused for every pasted point, and NPE training data
# is effectively unlimited from a small number of pastes.
#
# Each nu is a field-level harmonic draw through the contract's exact estimator.  A
# Cholesky draw in bandpower space is forbidden for this experiment: it bypasses the
# mask coupling and cannot be validated against map products.
#
# CPU only.  Measured 0.44 draws/s per worker; 2048 draws on 8 workers is ~10 min.

set -euo pipefail
REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
cd "$REPO"

$PY notebooks/SBI_validate/05_pasting/build_mock_sbi_noise_bank.py \
    --count 2048 --namespace training --workers 8 --threads-per-worker 2
$PY notebooks/SBI_validate/05_pasting/build_mock_sbi_noise_bank.py \
    --count 512 --namespace holdout --workers 8 --threads-per-worker 2

echo "=== calibration reports ==="
cat data/SBI_validate/mock_sbi/noise_bank_training.report.json
cat data/SBI_validate/mock_sbi/noise_bank_holdout.report.json
