#!/bin/bash
#SBATCH --job-name=sbi-cl-validate
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=data/SBI_validate/three_probe_mock/logs/noiseless_cls_%j.out
#SBATCH --error=data/SBI_validate/three_probe_mock/logs/noiseless_cls_%j.err

set -euo pipefail

REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PYTHON=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
CONFIG=${CONFIG:-${REPO}/notebooks/SBI_validate/three_probe_mock_experiment.yaml}
MAP=${MAP:-${REPO}/data/SBI_validate/three_probe_mock/maps/c0000_z0p3_0p5_mmin5e11_cosh32_fast/abacus_pasted_maps_c0000_z0p3_0p5_mmin5e11_nside512.h5}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO}/data/SBI_validate/three_probe_mock/validation/noiseless_cls_gate3}

cd "${REPO}"
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl-sbi-cl-${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}" "${OUTPUT_DIR}"

"${PYTHON}" -c 'import jax, pymaster, healpy; print("devices=", jax.devices()); print("pymaster=", pymaster.__version__); print("healpy=", healpy.__version__)'
"${PYTHON}" notebooks/SBI_validate/06_paste_vs_theory/compare_three_probe_noiseless_cls.py \
  --config "${CONFIG}" \
  --map "${MAP}" \
  --output-dir "${OUTPUT_DIR}"
