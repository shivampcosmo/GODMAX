#!/usr/bin/env bash
#SBATCH --job-name=des_tsz_zle0p85
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=256G
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/DES_cluster/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/DES_cluster/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
SCRIPT_DIR=${REPO}/notebooks/DES_cluster
PARAMS=${SCRIPT_DIR}/params_tsz_zmax0p85.yaml
DRIVER=${SCRIPT_DIR}/run_tsz_job.py
ENV_PREFIX=/mnt/home/spandey/ceph/env/godmax-jaxhealpy
ENV_HOOK=${ENV_PREFIX}/etc/conda/activate.d/godmax-jaxhealpy.sh
PYTHON=${ENV_PREFIX}/bin/python
EXPECTED_CONFIG_SHA256=30b918e83963d5ec16d6865654a5d0ba7b25742e2b9dba6637ee77985e221fed
EXPECTED_SELECTED_ROW_SHA256=3d702ceed8fb737ffcd31670792ae650fa11ddcdc0507389bac8e9627fab57b4
EXPECTED_HALOS=1299336

verify_sha256() {
  local expected=$1
  local file=$2
  local actual ignored
  read -r actual ignored < <(sha256sum "${file}")
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA256 contract mismatch for ${file}: ${actual} != ${expected}" >&2
    exit 2
  fi
}

verify_sha256 b6d4341eacb285d1319a704998cb011e9abcc34dfee4b566c4f75d1bba0ebec0 "${SCRIPT_DIR}/tsz_pasting.py"
verify_sha256 dd742af99c6e9b29918a81c912fbfcbf9b3da49233f10142e9ea5c50988025f4 "${DRIVER}"
verify_sha256 e7e7a87403790923b49d4da88fbc99047563973e771584b09c4e41cfd632abc8 "${SCRIPT_DIR}/params_tsz.yaml"
verify_sha256 93be0bc646dfc02fa1683ccdb134c6d17d9c986f82f9f57c387b22c27ecce6e7 "${PARAMS}"
verify_sha256 7473dbeb4345cc2b04c927180839c6f754079250212f390ab2cb8a7fb9169ba0 "${REPO}/notebooks/xDESI/abacus_pasting_helpers.py"
verify_sha256 b2fc086b530ec40105c4da84011b6bf8ffe422e1416c55b5e38d69fc9f9a6ad8 "${REPO}/src/get_sim_maps.py"
verify_sha256 f3012fecf58b15146010b7ec566b9f8284cad37efa78683ecf851e27cad95b5c "${REPO}/notebooks/xDESI/baryonforge_compare/matched_godmax_profiles.py"
verify_sha256 19d8cd36ff18b7517f069e4134a72d03378fa1000217951526cea1aacf61009b "${REPO}/param_files/params_default.yaml"
verify_sha256 6d66c203cd0207c81e4223729aace207e2e46b87f36ce4a78c4ba8ddd1d5cb96 "/mnt/ceph/users/spandey/xdesi/des/clusters/halo_lightcone_base_c000_ph000_Mgt1e13.h5"

if [[ -f "${ENV_HOOK}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_HOOK}"
fi

export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cuda
export JAX_PLATFORM_NAME=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib_${USER}_${SLURM_JOB_ID}}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${SCRIPT_DIR}/slurm_logs" "${MPLCONFIGDIR}"
cd "${REPO}"

echo "[tsz-submit] host=$(hostname) job=${SLURM_JOB_ID}"
echo "[tsz-submit] params=${PARAMS} python=${PYTHON}"
echo "[tsz-submit] cuda=${CUDA_VISIBLE_DEVICES:-unset} cpus=${SLURM_CPUS_PER_TASK:-unset}"
sha256sum "${SCRIPT_DIR}/tsz_pasting.py" "${PARAMS}" "${DRIVER}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

srun --ntasks=1 --nodes=1 --gres=gpu:1 --cpus-per-task=1 --cpu-bind=none \
  "${PYTHON}" -c 'import jax; jax.config.update("jax_enable_x64", True); print({"jax":jax.__version__,"backend":jax.default_backend(),"devices":[str(x) for x in jax.devices()],"x64":bool(jax.config.jax_enable_x64)}); assert jax.default_backend()=="gpu" and jax.config.jax_enable_x64'

/usr/bin/time -v srun --ntasks=1 --nodes=1 --gres=gpu:1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=none \
  "${PYTHON}" "${DRIVER}" --params "${PARAMS}" \
  --expected-config-sha256 "${EXPECTED_CONFIG_SHA256}" \
  --expected-selected-row-sha256 "${EXPECTED_SELECTED_ROW_SHA256}" \
  --expected-halos "${EXPECTED_HALOS}"
