#!/usr/bin/env bash
#SBATCH --job-name=bf_gm_stat2400
#SBATCH --partition=cmbas
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/baryonforge_compare/slurm_logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/baryonforge_compare/slurm_logs/%x.%j.err

set -euo pipefail
umask 027

readonly REPO_ROOT="/mnt/ceph/users/spandey/ltu-godmax/GODMAX"
readonly PYTHON="/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python"
readonly CONFIG="${REPO_ROOT}/notebooks/xDESI/baryonforge_compare/backlight_compare_projection_matched_cap2400.yaml"
readonly MEASURER="${REPO_ROOT}/notebooks/xDESI/baryonforge_compare/measure_statistics.py"
readonly PLOTTER="${REPO_ROOT}/notebooks/xDESI/baryonforge_compare/plot_results.py"
readonly RUN_ROOT="${REPO_ROOT}/data/xDESI/processed/baryonforge_godmax_backlight/mgt13_pz3_cap2400_projection_matched"
readonly GODMAX_MAP="${RUN_ROOT}/maps/godmax_projection_matched_mgt13_pz3_cap2400/abacus_pasted_maps_buffered_mgt13_nside1024_split000of001.h5"
readonly BARYONFORGE_MAP="${RUN_ROOT}/maps/baryonforge_native_nside1024.h5"
readonly STATISTICS="${RUN_ROOT}/measurements/common_mask_statistics.h5"
readonly PLOT_DIR="${RUN_ROOT}/plots"

readonly PLOT_CL_PNG="${PLOT_DIR}/01_all_cls.png"
readonly PLOT_CL_PDF="${PLOT_DIR}/01_all_cls.pdf"
readonly PLOT_DIAGNOSTIC_PNG="${PLOT_DIR}/02_cl_diagnostics.png"
readonly PLOT_DIAGNOSTIC_PDF="${PLOT_DIR}/02_cl_diagnostics.pdf"
readonly PLOT_Y_PNG="${PLOT_DIR}/03_tsz_maps.png"
readonly PLOT_Y_PDF="${PLOT_DIR}/03_tsz_maps.pdf"
readonly PLOT_KAPPA_PNG="${PLOT_DIR}/04_cmb_lensing_maps.png"
readonly PLOT_KAPPA_PDF="${PLOT_DIR}/04_cmb_lensing_maps.pdf"
readonly PLOT_MANIFEST="${PLOT_DIR}/plot_manifest.json"

export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/bf_gm_stat2400_matplotlib_${SLURM_JOB_ID}"

mkdir -p "${MPLCONFIGDIR}"
cd "${REPO_ROOT}"
echo "[$(date --iso-8601=seconds)] host=$(hostname) job_id=${SLURM_JOB_ID}"
echo "[$(date --iso-8601=seconds)] config=${CONFIG}"
echo "[$(date --iso-8601=seconds)] statistics=${STATISTICS}"
echo "[$(date --iso-8601=seconds)] plot_dir=${PLOT_DIR}"
echo "[$(date --iso-8601=seconds)] git_commit=$(git rev-parse --verify HEAD)"

test -s "${GODMAX_MAP}"
test -s "${BARYONFORGE_MAP}"
expected_outputs=(
  "${STATISTICS}"
  "${PLOT_CL_PNG}" "${PLOT_CL_PDF}"
  "${PLOT_DIAGNOSTIC_PNG}" "${PLOT_DIAGNOSTIC_PDF}"
  "${PLOT_Y_PNG}" "${PLOT_Y_PDF}"
  "${PLOT_KAPPA_PNG}" "${PLOT_KAPPA_PDF}"
  "${PLOT_MANIFEST}"
)
for output in "${expected_outputs[@]}"; do
  if [[ -e "${output}" ]]; then
    echo "Refusing to overwrite statistics/plot product: ${output}" >&2
    exit 64
  fi
done
sha256sum "${CONFIG}" "${MEASURER}" "${PLOTTER}" "${GODMAX_MAP}" "${BARYONFORGE_MAP}"

"${PYTHON}" -u "${MEASURER}" \
  --config "${CONFIG}" \
  --godmax-maps "${GODMAX_MAP}" \
  --baryonforge-maps "${BARYONFORGE_MAP}" \
  --output "${STATISTICS}"
test -s "${STATISTICS}"

"${PYTHON}" -u "${PLOTTER}" \
  --config "${CONFIG}" \
  --statistics "${STATISTICS}" \
  --godmax-maps "${GODMAX_MAP}" \
  --baryonforge-maps "${BARYONFORGE_MAP}" \
  --output-dir "${PLOT_DIR}" \
  --map-xsize 1000

for output in "${expected_outputs[@]}"; do
  test -s "${output}"
done
"${PYTHON}" -c 'import h5py,json,sys; stats,manifest=sys.argv[1:]; f=h5py.File(stats,"r"); assert str(f.attrs["schema"])=="baryonforge_godmax_common_mask_statistics_v1"; assert len(f["spectra"])==13; print({"n_spectra":len(f["spectra"]),"n_bins":len(f["ell"]),"fsky_binary":float(f["mask"].attrs["fsky_binary"])}); f.close(); m=json.load(open(manifest)); assert m["schema"]=="baryonforge_godmax_production_plots_v1"; assert len(m["spectrum_names"])==13; assert len(m["figures"])==8' "${STATISTICS}" "${PLOT_MANIFEST}"
sha256sum "${expected_outputs[@]}"
echo "[$(date --iso-8601=seconds)] completed statistics=${STATISTICS} plots=${PLOT_DIR}"
