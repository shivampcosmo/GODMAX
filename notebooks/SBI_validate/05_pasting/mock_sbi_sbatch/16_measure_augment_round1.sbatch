#!/bin/bash
#SBATCH --job-name=msbi_aug_r1
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/aug_r1_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/aug_r1_%j.err
#
# Measure the 128 combined maps into 42-vectors, then build the noise-augmented products.
#
# CPU only; no GPU is requested anywhere here, so the 32-GPU ceiling is untouched.
#
# Step 1 measures each combined map through the FROZEN estimator -- the regenerated float64
# mask, the hash-checked NaMaster workspace, the frozen galaxy alm, the 14 inference bands.
# It also runs two guards that exist because their failure is silent: no two distinct
# parameter points may share a map or produce an identical vector (injectivity), and every
# non-anchor point must differ from the reference paste by more than float noise, which is
# what a silently-unapplied gas override would look like.
#
# Step 2 adds noise. Per probe, the realization is drawn from that probe's own frozen
# spectrum and pushed through the real estimator:
#     y     : SO LAT deproj-2 component-separation noise PLUS the missing-sky yy term
#             Cyy(all mass, all z) - Cyy(pasted slice), smoothed by the same half-pixel
#             Bell as the painted maps
#     kappa : SO iterative minimum-variance CMB lensing reconstruction noise
#     tau   : the frozen SNR-matched white depth, 0.0233 tau-arcmin (provisional by
#             contract, matched to gkappa's 14-band amplitude forecast)
# Each draw is synalm -> alm2map -> mask-weighted centring -> map2alm -> alm2cl ->
# decouple_cell on the real mask. It is NOT an L@epsilon draw in bandpower space, which is
# forbidden for this experiment because it bypasses the mask coupling and the estimator.
#
# --write-maps 5 writes 5 float32 signal+noise map sets per point (y, tau, kappa_cmb),
# from the SAME seeds as the first 5 training rows, so a saved map and its row are the same
# realization. --replicas 64 is what the Stage-0 oracle validated; rows 5..63 cost 2.28 s
# each and no disk. The galaxy map is frozen across the five sampled gas parameters and is
# referenced by hash rather than copied 640 times.

set -euo pipefail
REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
PLAN=${PLAN:-$REPO/data/SBI_validate/mock_sbi/paste_plan/round_1/paste_plan.json}
REPLICAS=${REPLICAS:-64}
WRITE_MAPS=${WRITE_MAPS:-5}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd "$REPO/notebooks/SBI_validate"

echo "### step 1: measure the combined maps  $(date +%H:%M:%S)"
$PY measure_mock_sbi_pastes.py --plan "$PLAN"

RESP=$(dirname "$PLAN")/responses.npz
[ -s "$RESP" ] || { echo "FATAL: $RESP was not written"; exit 1; }

echo "### step 2: augment and write $WRITE_MAPS float32 map sets per point  $(date +%H:%M:%S)"
$PY build_mock_sbi_training_set.py \
    --responses "$RESP" \
    --output "$(dirname "$PLAN")/training_set_round1.npz" \
    --replicas "$REPLICAS" --write-maps "$WRITE_MAPS" --verify-maps 2

echo "### done  $(date +%H:%M:%S)"
