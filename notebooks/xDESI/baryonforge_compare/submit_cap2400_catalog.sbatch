#!/usr/bin/env bash
#SBATCH --job-name=bf_gm_cat2400
#SBATCH --partition=cmbas
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/baryonforge_compare/slurm_logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/baryonforge_compare/slurm_logs/%x.%j.err

set -euo pipefail
umask 027

readonly REPO_ROOT="/mnt/ceph/users/spandey/ltu-godmax/GODMAX"
readonly PYTHON="/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python"
readonly CONFIG="${REPO_ROOT}/notebooks/xDESI/baryonforge_compare/backlight_compare_projection_matched_cap2400.yaml"
readonly PREPARER="${REPO_ROOT}/notebooks/xDESI/baryonforge_compare/prepare_catalog.py"
readonly VALIDATOR="${REPO_ROOT}/notebooks/xDESI/baryonforge_compare/validate_config.py"
readonly PARENT_CATALOG="${REPO_ROOT}/data/xDESI/processed/abacus_backlight/stage31_pz3_cap2400_hmcbestfit_mmin11p147538/halos/abacus_c9999_ph9999_pz3cap2400_hmcbestfit_z0p63_0p98_logMgt11p147538_halos.h5"
readonly EXPECTED_PARENT_SHA256="481c49b4ff43f6f3d8822a34dc2af010d8450e594c3cc154f355fd7f3fcb33d6"
readonly EXPECTED_OUTPUT="${REPO_ROOT}/data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap2400/halos/abacus_c9999_ph9999_pz3cap2400_buffered_M200cgt1e13_halos.h5"
readonly NATIVE_CATALOG="${REPO_ROOT}/data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap2400_projection_matched/halos/abacus_c9999_ph9999_pz3cap2400_buffered_M200cgt1e13_halos.h5"

export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${REPO_ROOT}"
echo "[$(date --iso-8601=seconds)] host=$(hostname) job_id=${SLURM_JOB_ID}"
echo "[$(date --iso-8601=seconds)] config=${CONFIG}"
echo "[$(date --iso-8601=seconds)] expected_output=${EXPECTED_OUTPUT}"
echo "[$(date --iso-8601=seconds)] git_commit=$(git rev-parse --verify HEAD)"

test -s "${PARENT_CATALOG}"
if [[ -e "${EXPECTED_OUTPUT}" || -L "${EXPECTED_OUTPUT}" ]]; then
  echo "Refusing to overwrite strict-M200c catalog: ${EXPECTED_OUTPUT}" >&2
  exit 64
fi
if [[ -e "${NATIVE_CATALOG}" || -L "${NATIVE_CATALOG}" ]]; then
  echo "Refusing to replace native catalog route: ${NATIVE_CATALOG}" >&2
  exit 64
fi

actual_parent_sha256="$(sha256sum "${PARENT_CATALOG}" | awk '{print $1}')"
if [[ "${actual_parent_sha256}" != "${EXPECTED_PARENT_SHA256}" ]]; then
  echo "Parent catalog SHA256 ${actual_parent_sha256} != ${EXPECTED_PARENT_SHA256}." >&2
  exit 65
fi
sha256sum "${CONFIG}" "${PREPARER}" "${VALIDATOR}" "${PARENT_CATALOG}"

"${PYTHON}" -u "${VALIDATOR}" --config "${CONFIG}" --chunk-rows 2000000
"${PYTHON}" -u "${PREPARER}" --config "${CONFIG}" --chunk-rows 2000000
test -s "${EXPECTED_OUTPUT}"
"${PYTHON}" -c 'import h5py,numpy as np,sys; p=sys.argv[1]; f=h5py.File(p,"r"); m=np.asarray(f["M200c_hMsun"][:],dtype=np.float64); assert m.size==504174; assert np.all(m>1.0e13); assert int(f.attrs["selection_rows"])==504174; assert str(f.attrs["selection_predicate"])=="M200c_hMsun > 1.0e13"; print({"selected_rows":int(m.size),"mass_min_hMsun":float(m.min()),"mass_max_hMsun":float(m.max())}); f.close()' "${EXPECTED_OUTPUT}"

mkdir -p "$(dirname "${NATIVE_CATALOG}")"
ln -s "${EXPECTED_OUTPUT}" "${NATIVE_CATALOG}"
resolved_catalog="$(readlink -e "${NATIVE_CATALOG}")"
if [[ "${resolved_catalog}" != "${EXPECTED_OUTPUT}" ]]; then
  echo "Native catalog route resolves to ${resolved_catalog}, expected ${EXPECTED_OUTPUT}." >&2
  exit 66
fi
sha256sum "${EXPECTED_OUTPUT}" "${NATIVE_CATALOG}"
echo "[$(date --iso-8601=seconds)] completed output=${EXPECTED_OUTPUT}"

