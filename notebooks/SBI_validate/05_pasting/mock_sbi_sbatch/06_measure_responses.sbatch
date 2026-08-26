#!/bin/bash
#SBATCH --job-name=msbi_measure
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/measure_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/measure_%j.err
#
# Measure every combined map in a plan into its 42-vector through the frozen
# estimator, and run the injectivity and non-degeneracy guards.  CPU only, ~1.5 s per
# paste.  Fails closed if any two design points share a map, if two distinct points
# produce the same vector, or if any non-anchor point is indistinguishable from the
# reference paste -- which is what a silently unapplied gas override looks like.
#
#   PLAN=.../paste_plan.json sbatch 06_measure_responses.sbatch

set -euo pipefail
REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
: "${PLAN:?set PLAN to the paste_plan.json}"
export OMP_NUM_THREADS=8
cd "$REPO/notebooks/SBI_validate"
$PY measure_mock_sbi_pastes.py --plan "$PLAN"
