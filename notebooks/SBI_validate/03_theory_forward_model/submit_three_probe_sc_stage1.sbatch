#!/bin/bash
# Stage 1 (self-consistent contract): pin the reference point and prove the
# forward model is backend-portable on the node that will do the sampling.
#
# The observation here is the forward model's OWN prediction at a fixed parameter
# point, with the production covariance, Cholesky and windows reused byte-for-byte.
# On the production pasted-map contract the chi-square floor is 161.09 for a
# nominal 42-5 = 37, because the analytic model cannot reproduce the pasted maps --
# it is 218.67 even at the parameters the maps were painted at.  Any HMC-versus-SBI
# disagreement there is entangled with model misspecification.  Here chi2 at the
# generating point is zero, so a disagreement can only be the inference machinery.
#
# Resources: 1 node, 1 H100, 1 h.  Measured on an A100: reference point ~7 min,
# parity ~3 min.
#SBATCH --job-name=three_probe_sc_stage1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:h100_pcie:1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=logs/three_probe_sc_stage1_%j.out
#SBATCH --error=logs/three_probe_sc_stage1_%j.err

set -euo pipefail
export JAX_ENABLE_X64=True
SITE=/mnt/home/spandey/miniconda3/envs/ili-sbi/lib/python3.10/site-packages
export PATH=${SITE}/nvidia/cuda_nvcc/bin:${PATH}
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${SITE}/nvidia/cuda_nvcc"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export OMP_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/mpl-three-probe-sc-stage1-${SLURM_JOB_ID}

cd /mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/SBI_validate
mkdir -p logs
D=../../data/SBI_validate/three_probe_inference
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
CT=$D/inference_contract_selfconsistent.yaml
REF=$D/reference_point_selfconsistent.json

echo "### 1/4 forward-model non-regression (production contract, frozen mock) ###"
$PY validate_three_probe_jax_hmc_forward.py --output-dir $D/sc_stage1_forward_nonregression

echo "### 2/4 the observation reproduces its own generator on THIS node ###"
$PY verify_three_probe_selfconsistent_contract.py --contract "$CT" \
    --output $D/sc_stage1_selfconsistency.json

echo "### 3/4 reference point on the self-consistent contract ###"
$PY build_three_probe_reference_point.py --contract "$CT" --output "$REF"

echo "### 4/4 CPU/GPU parity, then stamp the verdict into the reference ###"
$PY check_three_probe_backend_parity.py --contract "$CT" --reference-point "$REF" \
    --output $D/sc_stage1_parity.json
$PY - "$REF" $D/sc_stage1_parity.json <<'PYX'
import json, pathlib, sys
ref_path, parity_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
reference = json.loads(ref_path.read_text())
parity = json.loads(parity_path.read_text())
reference["backend_portable"] = bool(parity["passed"])
ref_path.write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: parity[k] for k in
                  ("passed", "vector_passed", "chi2_passed",
                   "max_relative_difference", "chi2_at_map_difference",
                   "this_device_kind")}, indent=2, sort_keys=True))
if not parity["passed"]:
    raise SystemExit(
        "Backend parity FAILED. Refusing to stamp a portable verdict: a posterior "
        "from a forward model that disagrees with itself across backends is not a "
        "result. Stages 2a and 2b will refuse to start.")
print("stamped backend_portable = True")
PYX
echo "### stage 1 complete ###"
