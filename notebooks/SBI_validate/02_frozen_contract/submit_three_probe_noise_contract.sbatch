#!/bin/bash
#SBATCH --job-name=sbi-noise-contract
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=data/SBI_validate/three_probe_mock/logs/noise_contract_%j.out
#SBATCH --error=data/SBI_validate/three_probe_mock/logs/noise_contract_%j.err

set -euo pipefail
REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PYTHON=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
ROOT=${REPO}/data/SBI_validate/three_probe_mock/validation/noisy_nside1024_ell2048
MAP=${REPO}/data/SBI_validate/three_probe_mock/maps/c0000_z0p3_0p5_mmin5e11_cosh32_fast/abacus_pasted_maps_c0000_z0p3_0p5_mmin5e11_nside1024.h5
cd "${REPO}"
mkdir -p "${ROOT}" "${REPO}/data/SBI_validate/three_probe_mock/logs"
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl-sbi-noise-${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}"
"${PYTHON}" -c 'import jax,pymaster,healpy; print("devices=",jax.devices()); print("pymaster=",pymaster.__version__); print("healpy=",healpy.__version__)'
"${PYTHON}" notebooks/SBI_validate/common/three_probe_noise_contract.py build \
  --config notebooks/SBI_validate/three_probe_mock_experiment.yaml \
  --map "${MAP}" \
  --output "${ROOT}/noise_contract.h5" \
  --workspace "${ROOT}/namaster_workspace.fits" \
  --n-projected-radius 256

