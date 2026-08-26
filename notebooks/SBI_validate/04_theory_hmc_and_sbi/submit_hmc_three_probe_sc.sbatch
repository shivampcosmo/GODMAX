#!/bin/bash
# Stage 2a (self-consistent contract): four-chain preconditioned NUTS.
#
# What changed from hmc_v2/run01, which returned COMPLETED_REJECTED with 72.24%
# tree-depth saturation, max r_hat 1.0127 and min ESS 712 in 8.9 h:
#
#  * the observation.  On the pasted-map contract the model strains against a
#    misfit it cannot absorb; the posterior sits on a stiff, non-Gaussian profile
#    (delta chi2 = 10 at one Laplace sigma, flattening to 14 at two) that forced a
#    step size of 0.02 and ~205 leapfrog steps per trajectory against a ceiling of
#    127.  Here chi2 at the generating point is zero.  Measured on an A100 trial:
#    3-5 s/it versus a flat 8.13 s/it before.
# target_accept STAYS at 0.90.  A trial at 0.80 confirmed saturation is gone but
# produced 3 divergences in 400 draws; 0.90 is the setting that gave ZERO
# divergences on the production run, and it is now affordable because trajectories
# cost ~33 steps instead of 107.  The mass matrix was never the problem and is
# unchanged: warm-up adaptation already took the metric from condition 3044
# against the sampled covariance down to 1.66-2.54.
#
# Sizing, from measured per-leapfrog-step cost rather than a guess:
#   A100 trial 0.1922 s/step, this partition's H100 PCIe 0.0756 s/step (2.54x).
#   1000 warm-up + 1800 draws at ~33 steps  ->  2.2 h expected on H100.
#   If the H100 behaved like the A100 it would be 5.4 h, so --max-wall-seconds
#   16200 (4.5 h) stops sampling cleanly and still writes a full verdict; that
#   floor is ~1200 draws/chain, ESS ~820 against a gate of 400.
# The wall budget is enforced by the runner, not by SLURM killing the process, so
# the run cannot end with only checkpoints and no gate.
#
# Resources: 1 node, 1 H100, 5 h.
#SBATCH --job-name=three_probe_hmc_sc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:h100_pcie:1
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=logs/three_probe_hmc_sc_%j.out
#SBATCH --error=logs/three_probe_hmc_sc_%j.err

set -euo pipefail
export JAX_ENABLE_X64=True
SITE=/mnt/home/spandey/miniconda3/envs/ili-sbi/lib/python3.10/site-packages
export PATH=${SITE}/nvidia/cuda_nvcc/bin:${PATH}
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${SITE}/nvidia/cuda_nvcc"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export OMP_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/mpl-three-probe-hmc-sc-${SLURM_JOB_ID}

cd /mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/SBI_validate
mkdir -p logs
D=../../data/SBI_validate/three_probe_inference
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
CT=$D/inference_contract_selfconsistent.yaml
REF=$D/reference_point_selfconsistent.json
OUT=$D/hmc_sc/run01
mkdir -p "$OUT"

$PY run_hmc_three_probe_v2.py --contract "$CT" --reference-point "$REF" \
    --output-dir "$OUT" --preflight-only
$PY run_hmc_three_probe_v2.py --contract "$CT" --reference-point "$REF" \
    --output-dir "$OUT" \
    --warmup 1000 --samples 1800 --chains 4 \
    --max-tree-depth 7 --target-accept 0.90 \
    --checkpoint-interval 100 --max-wall-seconds 16200 --resume
