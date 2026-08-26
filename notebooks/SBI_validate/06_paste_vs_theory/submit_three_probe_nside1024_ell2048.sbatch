#!/bin/bash
#SBATCH --job-name=sbi-cl-n1024-l2048
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=data/SBI_validate/three_probe_mock/logs/nside1024_ell2048_%j.out
#SBATCH --error=data/SBI_validate/three_probe_mock/logs/nside1024_ell2048_%j.err

set -euo pipefail
REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PYTHON=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
MAP=${REPO}/data/SBI_validate/three_probe_mock/maps/c0000_z0p3_0p5_mmin5e11_cosh32_fast/abacus_pasted_maps_c0000_z0p3_0p5_mmin5e11_nside1024.h5
OLD=${REPO}/data/SBI_validate/three_probe_mock/validation/noiseless_cls_gate3_nside1024/first_matched_current_paste_vs_projected_theory.h5
OUT=${REPO}/data/SBI_validate/three_probe_mock/validation/noiseless_cls_gate3_nside1024/ell2048
cd "${REPO}"
export JAX_ENABLE_X64=True JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl-sbi-n1024-l2048-${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}"
"${PYTHON}" -c 'import jax,pymaster,healpy; print("devices=",jax.devices()); print("pymaster=",pymaster.__version__); print("healpy=",healpy.__version__)'
"${PYTHON}" notebooks/SBI_validate/06_paste_vs_theory/compare_three_probe_nside1024_ell2048.py \
  --config notebooks/SBI_validate/three_probe_mock_experiment.yaml \
  --map "${MAP}" \
  --old-artifact "${OLD}" \
  --output-dir "${OUT}"
