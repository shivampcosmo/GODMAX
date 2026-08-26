#!/bin/bash
# Canary job: cheap-grid reference point, then an end-to-end smoke of BOTH
# runners.  Runs in parallel with the reference job and gates only the SBI
# production run, because the SBI path is the one still carrying newly written
# code (the round-2 proposal handover).  The HMC production run deliberately
# does NOT depend on this, so an SBI fault can no longer block it.
#
# Resources: 1 node, 1 H100, 1 h wall.
#SBATCH --job-name=three_probe_smoke_v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:h100_pcie:1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=logs/three_probe_smoke_v2_%j.out
#SBATCH --error=logs/three_probe_smoke_v2_%j.err

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
export MPLCONFIGDIR=/tmp/mpl-three-probe-smoke-${SLURM_JOB_ID}

cd /mnt/ceph/users/spandey/ltu-godmax/GODMAX
mkdir -p logs
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
SRC=notebooks/SBI_validate
SMOKE=data/SBI_validate/three_probe_inference/smoke_v2/job_${SLURM_JOB_ID}
mkdir -p "${SMOKE}"
export SMOKE_ABS=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/${SMOKE}
cd "${SRC}"

echo "=== [1/3] cheap-grid reference point ==="
$PY build_three_probe_reference_point.py \
  --grid-override 64 48 22 64 --maxiter 300 --max-gradient-norm 0.1 \
  --output "../../${SMOKE}/reference_point_smoke.json"
# The cheap smoke grid's coarse quadrature puts a noise floor of ~3e-2 on the
# gradient norm, so the production gate of 1e-2 is unreachable here however long
# L-BFGS runs -- job 6928688 stopped at |grad| = 3.471e-02 with 6 of 8 starts
# agreeing to 3.0e-03 in u and 1.8e-07 in potential, i.e. genuinely converged for
# this grid.  0.1 still catches the failure this gate exists for: the zero-gradient
# regression gave |grad| = 1.5e+01 and 2.6e+01, rejected by 150x and 260x.  The
# PRODUCTION gate is untouched at 1e-2, where the converged grid measures 3.4e-04.

# Exit code 3 means "ran to completion, failed a blind gate", which is the
# expected outcome for a 512-simulation smoke run.  Any other non-zero code is a
# real fault and must stop the chain before the production jobs are released.
echo "=== [2/3] HMC runner smoke ==="
set +e
$PY run_hmc_three_probe_v2.py \
  --output-dir "../../${SMOKE}/hmc" \
  --reference-point "../../${SMOKE}/reference_point_smoke.json" \
  --warmup 25 --samples 50 --chains 4 --max-tree-depth 4 --replay-draws 8 \
  --checkpoint-interval 25 --smoke-test
status=$?
set -e
if [ "$status" -ne 0 ] && [ "$status" -ne 3 ]; then echo "HMC smoke FAILED ($status)"; exit "$status"; fi
echo "HMC smoke completed (exit $status)"

echo "=== [3/3] SBI runner smoke ==="
set +e
$PY run_sbi_three_probe_v2.py \
  --output-dir "../../${SMOKE}/sbi" \
  --reference-point "../../${SMOKE}/reference_point_smoke.json" \
  --rounds-override 512 512 --posterior-draws 2000 --validation-draws 512 --smoke-test
status=$?
set -e
if [ "$status" -ne 0 ] && [ "$status" -ne 3 ]; then echo "SBI smoke FAILED ($status)"; exit "$status"; fi
echo "SBI smoke completed (exit $status)"

echo "=== smoke chain complete: both runners ran end to end ==="
