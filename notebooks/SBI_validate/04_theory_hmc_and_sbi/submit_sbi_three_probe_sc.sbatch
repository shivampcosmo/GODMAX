#!/bin/bash
# Stage 2b (self-consistent contract): score-compressed sequential NPE with an
# in-job exact-likelihood cross-check.  Runs beside stage 2a, never behind it.
#
# What changed from sbi_v2/run01, which returned COMPLETED_REJECTED:
#
#  * the observation.  The score summary of a noiseless self-consistent
#    observation is |s| = 0.241 against a typical draw of 1.743, i.e. the
#    conditioning point is now near the centre of the simulator's own
#    distribution instead of a ~13 sigma outlier.
#  * simulation budget 131,072 -> 262,144 (4 rounds x 65,536), as requested.
#  * the exact-likelihood importance proposal.  A two-component elliptical
#    mixture gave Pareto k = 0.935 and ESS 25 of 20,000 -- the independent
#    reference could not adjudicate the thing it exists for.  It is now a
#    k-means Student-t mixture (MAP Laplace + an over-dispersed safety net + up
#    to 8 clusters of the NPE draws), which follows an arbitrary posterior shape
#    and degrades gracefully.  Measured on a banana target: ESS 7724 versus 2693,
#    max weight 0.0006 versus 0.0044.
#  * the identifiable-region truncation from the previous fix is retained; on the
#    pasted-map contract round 4's proposal acceptance inside |u| <= 8 was 0.446,
#    so without it the run dies as job 6928795 did.
#
# Sizing, from the measured trial (3 x 16,384 in 31 min end to end): simulation is
# ~8 min per 65,536-round and MDN training dominates and grows with the cumulative
# set, giving ~2.6 h for four rounds.  --max-wall-seconds 16200 (4.5 h) declines to
# START a round that would overrun and goes straight to the exact-likelihood
# validation, so a slow training pass costs campaign length rather than the whole
# result.  A round cannot be stopped part-way, which is why the check is per round.
#
# Resources: 1 node, 1 H100, 5 h.
#SBATCH --job-name=three_probe_sbi_sc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:h100_pcie:1
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=logs/three_probe_sbi_sc_%j.out
#SBATCH --error=logs/three_probe_sbi_sc_%j.err

set -euo pipefail
export JAX_ENABLE_X64=True
SITE=/mnt/home/spandey/miniconda3/envs/ili-sbi/lib/python3.10/site-packages
export PATH=${SITE}/nvidia/cuda_nvcc/bin:${PATH}
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${SITE}/nvidia/cuda_nvcc"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export OMP_NUM_THREADS=14
export MPLCONFIGDIR=/tmp/mpl-three-probe-sbi-sc-${SLURM_JOB_ID}

cd /mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/SBI_validate
mkdir -p logs
D=../../data/SBI_validate/three_probe_inference
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
CT=$D/inference_contract_selfconsistent.yaml
REF=$D/reference_point_selfconsistent.json
OUT=$D/sbi_sc/run01
mkdir -p "$OUT"

$PY run_sbi_three_probe_v2.py --contract "$CT" --reference-point "$REF" \
    --output-dir "$OUT" --forward-batch-size 16 --resume --preflight-only
$PY run_sbi_three_probe_v2.py --contract "$CT" --reference-point "$REF" \
    --output-dir "$OUT" --forward-batch-size 16 --max-wall-seconds 16200 --resume
