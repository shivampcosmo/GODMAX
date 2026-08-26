#!/bin/bash
# Stage 2a: preconditioned four-chain NUTS on the converged grid.
#
# Depends on stage 1 (reference_point_v2.json).  Changes versus the rejected
# depth-6 run: probit coordinates, Laplace inverse-mass matrix and MAP start,
# tree depth 7 (capped), 1500 warm-up steps, converged forward grid, full src/
# source manifest, recorded backend, and an in-artifact chi-square replay.
#
# Depends ONLY on the reference-point job, never on the SBI smoke, so an SBI
# fault cannot block it.  Runs side by side with the SBI job on a separate GPU.
#
# Resources: 1 node, 1 H100, 36 h wall.  Measured from the smoke: 0.0947 s per
# leapfrog step for 4 vectorized chains on the coarse grid, and the converged
# grid costs 2.21x, giving 0.209 s.  Over 1500 + 2500 = 4000 iterations that is
# 4.6 h at a mean of 20 steps per draw, 13.9 h at 60, and 29.5 h if trajectories
# saturate at depth 7's 127 steps.  Checkpointing plus --resume makes an
# overrun survivable: requeue this script and it continues.
#SBATCH --job-name=three_probe_hmc_v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:h100_pcie:1
#SBATCH --mem=120G
#SBATCH --time=36:00:00
#SBATCH --output=logs/three_probe_hmc_v2_%j.out
#SBATCH --error=logs/three_probe_hmc_v2_%j.err

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
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export MPLCONFIGDIR=/tmp/mpl-three-probe-hmc-v2-${SLURM_JOB_ID}

cd /mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/SBI_validate
mkdir -p ../../logs
# Fixed output directory so a requeue resumes the same chain rather than
# starting a fresh one under a new job id.
OUT=../../data/SBI_validate/three_probe_inference/hmc_v2/run01
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python

# --resume is safe on a first run: it is a no-op when no complete checkpoint
# exists, so requeueing this script after a wall-clock kill continues rather
# than restarting.  Checkpoints land in ${OUT}/checkpoint_NNNNNN.*, and
# ${OUT}/progress.json carries the live status.
$PY run_hmc_three_probe_v2.py --output-dir "${OUT}" --preflight-only
$PY run_hmc_three_probe_v2.py --output-dir "${OUT}" \
  --warmup 1500 --samples 2500 --chains 4 --max-tree-depth 7 --target-accept 0.9 \
  --checkpoint-interval 100 --resume
