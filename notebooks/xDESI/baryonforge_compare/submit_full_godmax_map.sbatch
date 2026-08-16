#!/usr/bin/env bash
#SBATCH --job-name=bf_gm_godmax_full
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/baryonforge_compare/slurm_logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/baryonforge_compare/slurm_logs/%x.%j.err

set -euo pipefail
umask 027

readonly REPO_ROOT="/mnt/ceph/users/spandey/ltu-godmax/GODMAX"
readonly PYTHON="/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python"
readonly CONFIG="${REPO_ROOT}/notebooks/xDESI/baryonforge_compare/backlight_compare_projection_matched.yaml"
readonly PAINTER="${REPO_ROOT}/notebooks/xDESI/baryonforge_compare/paint_godmax.py"
readonly EXPECTED_OUTPUT="${REPO_ROOT}/data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap600_projection_matched/maps/godmax_projection_matched_mgt13_pz3_cap600/abacus_pasted_maps_buffered_mgt13_nside1024_split000of001.h5"
readonly CONTRACT_CATALOG="${REPO_ROOT}/data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap600/halos/abacus_c9999_ph9999_pz3cap600_buffered_M200cgt1e13_halos.h5"
readonly NATIVE_CATALOG="${REPO_ROOT}/data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap600_projection_matched/halos/abacus_c9999_ph9999_pz3cap600_buffered_M200cgt1e13_halos.h5"

export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cuda
export PASTE_JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/bf_gm_matplotlib_${SLURM_JOB_ID}"

if [[ -n "${CRAY_CUDATOOLKIT_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="/mnt/home/spandey/miniconda3/envs/ili-sbi/lib:${CRAY_CUDATOOLKIT_DIR}/lib64:${LD_LIBRARY_PATH:-}"
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_cuda_data_dir=${CRAY_CUDATOOLKIT_DIR}"
fi

mkdir -p "${MPLCONFIGDIR}"
cd "${REPO_ROOT}"

echo "[$(date --iso-8601=seconds)] host=$(hostname) job_id=${SLURM_JOB_ID}"
echo "[$(date --iso-8601=seconds)] python=${PYTHON}"
echo "[$(date --iso-8601=seconds)] config=${CONFIG}"
echo "[$(date --iso-8601=seconds)] expected_output=${EXPECTED_OUTPUT}"
echo "[$(date --iso-8601=seconds)] git_commit=$(git rev-parse --verify HEAD)"
sha256sum "${CONFIG}" "${PAINTER}"
resolved_catalog="$(readlink -f "${NATIVE_CATALOG}")"
if [[ "${resolved_catalog}" != "${CONTRACT_CATALOG}" ]]; then
  echo "Native catalog route resolves to ${resolved_catalog}, expected ${CONTRACT_CATALOG}." >&2
  exit 65
fi
sha256sum "${NATIVE_CATALOG}"
if [[ -e "${EXPECTED_OUTPUT}" ]]; then
  echo "Refusing to overwrite existing GODMAX output: ${EXPECTED_OUTPUT}" >&2
  exit 64
fi
nvidia-smi
"${PYTHON}" -c 'import jax, jax.numpy as jnp, jaxlib; result = jax.jit(lambda value: value + 1.0)(jnp.asarray(1.0)); result.block_until_ready(); print({"jax": jax.__version__, "jaxlib": jaxlib.__version__, "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()], "compiled_result": float(result)})'

"${PYTHON}" -u "${PAINTER}" \
  --config "${CONFIG}" \
  --pixel-workers 1

test -s "${EXPECTED_OUTPUT}"
echo "[$(date --iso-8601=seconds)] completed output=${EXPECTED_OUTPUT}"
