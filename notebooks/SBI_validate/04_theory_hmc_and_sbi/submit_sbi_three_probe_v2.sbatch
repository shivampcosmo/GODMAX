#!/bin/bash
# Stage 2b: score-compressed sequential NPE plus an in-job exact-likelihood
# cross-check, on the same converged grid and the same pinned reference point as
# stage 2a.
#
# The decisive change is the summary: the exact 5-dimensional normalized score at
# the pinned MAP, which projects out the 37-dimensional subspace carrying the
# absolute misfit and makes the observation a typical simulator draw.  The
# previous campaign conditioned an MDN on the raw 42-vector at a ~13-sigma
# outlier, which no architecture, round count or PCA could fix.
#
# Resources: 1 node, 1 H100, 8 h wall.  98,304 forward simulations plus 20,000
# exact-likelihood evaluations; the 65,536-simulation run took 675 s on the
# coarse grid, and the converged grid costs 2.2x.
#SBATCH --job-name=three_probe_sbi_v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:h100_pcie:1
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/three_probe_sbi_v2_%j.out
#SBATCH --error=logs/three_probe_sbi_v2_%j.err

set -euo pipefail
export JAX_ENABLE_X64=True

# Compiler toolchain.  Job 6928453 established that this makes NO difference:
# ptxas 12.8.61 reproduces the module's 12.5.82 bit-for-bit on this forward
# model.  It is kept only to silence JAX's "CUDA <=12.6.2 miscompiles
# clamping" warning, which is a proven red herring here, so that nobody
# chases it again.  The CPU/GPU divergence has another cause and the
# stage-1 parity gate below is what actually protects against it.
SITE=/mnt/home/spandey/miniconda3/envs/ili-sbi/lib/python3.10/site-packages
export PATH=${SITE}/nvidia/cuda_nvcc/bin:${PATH}
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${SITE}/nvidia/cuda_nvcc"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export OMP_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/mpl-three-probe-sbi-v2-${SLURM_JOB_ID}

cd /mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/SBI_validate
mkdir -p ../../logs
OUT=../../data/SBI_validate/three_probe_inference/sbi_v2/run01
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python

$PY run_sbi_three_probe_v2.py --output-dir "${OUT}" --forward-batch-size 16 --resume --preflight-only
$PY run_sbi_three_probe_v2.py --output-dir "${OUT}" --forward-batch-size 16 --resume
