#!/usr/bin/env bash
#SBATCH --job-name=bf_gm_bf2400
#SBATCH --partition=cmbas
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/baryonforge_compare/slurm_logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/baryonforge_compare/slurm_logs/%x.%j.err

set -euo pipefail
umask 027

readonly REPO_ROOT="/mnt/ceph/users/spandey/ltu-godmax/GODMAX"
readonly PYTHON="/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python"
readonly CONFIG="${REPO_ROOT}/notebooks/xDESI/baryonforge_compare/backlight_compare_projection_matched_cap2400.yaml"
readonly PAINTER="${REPO_ROOT}/notebooks/xDESI/baryonforge_compare/paint_baryonforge.py"
readonly CATALOG="${REPO_ROOT}/data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap2400/halos/abacus_c9999_ph9999_pz3cap2400_buffered_M200cgt1e13_halos.h5"
readonly EXPECTED_OUTPUT="${REPO_ROOT}/data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap2400_projection_matched/maps/baryonforge_native_nside1024.h5"

export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/bf_gm_bf2400_matplotlib_${SLURM_JOB_ID}"

mkdir -p "${MPLCONFIGDIR}"
cd "${REPO_ROOT}"
echo "[$(date --iso-8601=seconds)] host=$(hostname) job_id=${SLURM_JOB_ID}"
echo "[$(date --iso-8601=seconds)] config=${CONFIG}"
echo "[$(date --iso-8601=seconds)] expected_output=${EXPECTED_OUTPUT}"
echo "[$(date --iso-8601=seconds)] git_commit=$(git rev-parse --verify HEAD)"

test -s "${CATALOG}"
if [[ -e "${EXPECTED_OUTPUT}" ]]; then
  echo "Refusing to overwrite BaryonForge output: ${EXPECTED_OUTPUT}" >&2
  exit 64
fi
sha256sum "${CONFIG}" "${PAINTER}" "${CATALOG}"

"${PYTHON}" -u "${PAINTER}" --config "${CONFIG}" --n-jobs 8
test -s "${EXPECTED_OUTPUT}"
"${PYTHON}" -c 'import h5py,sys; f=h5py.File(sys.argv[1],"r"); assert str(f.attrs["backend"])=="baryonforge"; assert int(f.attrs["halo_count"])==504174; assert int(f.attrs["n_halos_painted"])==504174; assert bool(f.attrs["complete_catalog_paint"]); assert int(f.attrs["nside"])==1024; assert not bool(f.attrs["smooth_profiles"]); print({"halo_count":int(f.attrs["halo_count"]),"nside":int(f.attrs["nside"]),"complete":bool(f.attrs["complete_catalog_paint"])}); f.close()' "${EXPECTED_OUTPUT}"
sha256sum "${EXPECTED_OUTPUT}"
echo "[$(date --iso-8601=seconds)] completed output=${EXPECTED_OUTPUT}"

