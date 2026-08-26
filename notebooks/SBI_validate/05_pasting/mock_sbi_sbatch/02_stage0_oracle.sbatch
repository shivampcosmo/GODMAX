#!/bin/bash
#SBATCH --job-name=msbi_oracle
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/oracle_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/oracle_%j.err
#
# Stage 0: decide the paste budget BEFORE spending any GPU time on pastes.
#
# The analytic theory model stands in for the expensive simulator (21 ms per
# evaluation), is emulated from N design points drawn by the frozen mixture recipe,
# and is judged on held-out likelihood error and on the posterior actually obtained
# through the emulator versus the archived exact NUTS chain.
#
# Two arms:
#   full        emulate the whole response.  Conservative: no help from the transfer
#               factorization, so whatever N passes here is an upper bound.
#   factorized  emulate the ratio to a cheap coarse-grid baseline and multiply it
#               back -- the production architecture.  The baseline is this project's
#               previous production grid, documented FAIL at 1e-2 median against the
#               converged grid, so the ratio has a realistic amplitude and real
#               theta-dependence rather than an invented synthetic form.
#
# Gate: the smallest N meeting p95|dchi2| <= 0.10, max <= 0.50, posterior mean drift
# <= 0.10 sigma, width change <= 10%, correlation change <= 0.10, over three emulator
# seeds, is the pre-registered round-3 target.  If NO N <= 384 passes in either arm,
# the architecture must change before any paste is submitted -- do not simply buy
# more pastes.

set -uo pipefail
REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
OUT=$REPO/data/SBI_validate/mock_sbi
export JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=/tmp/matplotlib
export OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore
cd "$REPO/notebooks/SBI_validate"

status=0
for arm in full factorized; do
  echo "################ arm=$arm $(date -Is)"
  $PY oracle_paste_budget_test.py \
      --n-design 512 --emcee-steps 3000 --skip-exact-emcee \
      --recipe concentrated --target "$arm" --output-dir "$OUT" || status=1
done
echo "### recommended N per arm"
grep -h '"recommended_n_train"' "$OUT"/oracle_budget_test_*.json || true
exit $status
