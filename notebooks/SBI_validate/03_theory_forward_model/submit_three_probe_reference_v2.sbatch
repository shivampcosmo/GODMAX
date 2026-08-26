#!/bin/bash
# Stage 1 of the three-probe HMC/SBI agreement campaign.
#
# Builds ONLY the pinned reference point (MAP, Laplace covariance, score
# operator, backend parity vectors) that both the HMC and the SBI job consume,
# plus the CPU-versus-GPU parity record.  The runner smoke tests deliberately
# live in a separate job (submit_three_probe_smoke_v2.sbatch) so that a fault in
# one runner cannot block the other's production run.
#
# Resources: 1 node, 1 H100, 2 h wall.  Produces:
#   data/SBI_validate/three_probe_inference/reference_point_v2.json
#SBATCH --job-name=three_probe_ref_v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:h100_pcie:1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/three_probe_ref_v2_%j.out
#SBATCH --error=logs/three_probe_ref_v2_%j.err

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
export MPLCONFIGDIR=/tmp/mpl-three-probe-ref-${SLURM_JOB_ID}

cd /mnt/ceph/users/spandey/ltu-godmax/GODMAX
mkdir -p logs
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
SRC=notebooks/SBI_validate
SMOKE=data/SBI_validate/three_probe_inference/reference_v2/job_${SLURM_JOB_ID}
mkdir -p "${SMOKE}"
export SMOKE_ABS=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/${SMOKE}

cd "${SRC}"

# The forward model's values changed when the float32 quantiser was corrected
# (denormals are now rounded onto the 2**-149 grid rather than flushed), so the
# frozen non-regression check against the NON-JAX host implementation runs first.
# It is the only independent check that the operator is still the right physics,
# it exits non-zero on failure, and it costs ~30 s.
echo "=== [1/3] forward model versus the non-JAX host implementation (0.5% gate) ==="
$PY validate_three_probe_jax_hmc_forward.py --output-dir "../../${SMOKE}/host_validation"

echo "=== [2/3] production reference point on the converged grid (256,48,48,2049) ==="
$PY build_three_probe_reference_point.py --maxiter 600
REF=../../data/SBI_validate/three_probe_inference/reference_point_v2.json
export REF_ABS=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/three_probe_inference/reference_point_v2.json

# Settle the CPU-versus-GPU question on this node, on the converged grid, with
# byte-identical sources.  A CPU replay of the depth-6 artifact disagreed with
# its own stored chi-square by ~88 units and the backend was the only remaining
# difference; this records the answer instead of leaving it inferred.
echo "=== [3/3] backend parity: GPU then CPU, same node, same reference ==="
$PY check_three_probe_backend_parity.py --reference-point "${REF}" \
   --output "../../${SMOKE}/backend_parity_gpu.json"
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=15 \
$PY check_three_probe_backend_parity.py --reference-point "${REF}" \
   --output "../../${SMOKE}/backend_parity_cpu.json"
$PY - <<'PYCHECK'
import json, pathlib, os
base = pathlib.Path(os.environ["SMOKE_ABS"])
gpu = json.loads((base / "backend_parity_gpu.json").read_text())
cpu = json.loads((base / "backend_parity_cpu.json").read_text())
print("\n=== CPU versus GPU forward model, identical sources, same node ===")
print(f"  sources identical      : {gpu['sources_identical']} / {cpu['sources_identical']}")
print(f"  chi2 at MAP on GPU     : {gpu['chi2_at_map_here']:.4f}")
print(f"  chi2 at MAP on CPU     : {cpu['chi2_at_map_here']:.4f}")
print(f"  difference             : {cpu['chi2_at_map_here'] - gpu['chi2_at_map_here']:+.4f}")
print(f"  max relative fwd diff  : GPU {gpu['max_relative_difference']:.3e}  "
      f"CPU {cpu['max_relative_difference']:.3e}  (tolerance {cpu['vector_relative_tolerance']:.0e})")
print(f"  chi2 difference        : {cpu['chi2_at_map_difference']:+.3e}  "
      f"(tolerance {cpu['chi2_tolerance']:.0e})")
print(f"  gates                  : vector {'PASS' if cpu['vector_passed'] else 'FAIL'}, "
      f"chi2 {'PASS' if cpu['chi2_passed'] else 'FAIL'}")
portable = bool(cpu["passed"])
print(f"  VERDICT                : forward model is "
      f"{'PORTABLE' if portable else 'NOT BACKEND-PORTABLE'}")
(base / "backend_portable.json").write_text(json.dumps(dict(
    portable=bool(portable),
    max_relative_difference=cpu["max_relative_difference"],
    chi2_at_map_gpu=gpu["chi2_at_map_here"], chi2_at_map_cpu=cpu["chi2_at_map_here"]),
    indent=2, sort_keys=True) + "\n")
ref_path = pathlib.Path(os.environ["REF_ABS"])
payload = json.loads(ref_path.read_text())
payload["backend_portable"] = bool(portable)
ref_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
if not portable:
    # Fail closed.  A posterior from a forward model that disagrees with itself
    # across backends is not a result, however well two samplers agree on it.
    raise SystemExit("Backend parity FAILED: refusing to release the production jobs")
PYCHECK

echo "=== stage 1 complete ==="
sha256sum "${REF}"
