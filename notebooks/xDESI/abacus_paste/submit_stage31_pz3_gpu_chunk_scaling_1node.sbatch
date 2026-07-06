#!/usr/bin/env bash
#SBATCH --job-name=pz3_gpu_scaling
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --mem=0
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
SCRIPT_DIR=${REPO}/notebooks/xDESI/abacus_paste
CONDA_SH=${CONDA_SH:-/mnt/home/spandey/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-godmax-jaxhealpy}
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-/mnt/home/spandey/ceph/env/godmax-jaxhealpy}
CONDA_ENV_HOOK=${CONDA_ENV_HOOK:-${CONDA_ENV_PREFIX}/etc/conda/activate.d/godmax-jaxhealpy.sh}
DRIVER=${SCRIPT_DIR}/stage31_pz1_backlight_validation.py
CONFIG=${CONFIG:-${SCRIPT_DIR}/stage31_pz3_cap600.selected.yaml}

NSIDE_LIST=${NSIDE_LIST:-512 1024}
NHALOS_LIST=${NHALOS_LIST:-2000 200000}
SAMPLE_MODE=${SAMPLE_MODE:-random}
PIXEL_WORKERS=${PIXEL_WORKERS:-16}
PIXEL_POOL_CHUNKSIZE=${PIXEL_POOL_CHUNKSIZE:-32}
PIXEL_START_METHOD=${PIXEL_START_METHOD:-fork}
PIXEL_BACKEND=${PIXEL_BACKEND:-healpy}
QUERY_DISC_BUFFER_SAFETY_FACTOR=${QUERY_DISC_BUFFER_SAFETY_FACTOR:-2.0}
INCLUDE_GALAXIES=${INCLUDE_GALAXIES:-0}
GALAXY_POPULATION_BACKEND=${GALAXY_POPULATION_BACKEND:-compact}
GALAXY_COMPACT_MAX_SATELLITE_GROUPS=${GALAXY_COMPACT_MAX_SATELLITE_GROUPS:-8}

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi
if [[ -f "${CONDA_ENV_HOOK}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_ENV_HOOK}"
fi
PYTHON=${PYTHON:-${CONDA_ENV_PREFIX}/bin/python}

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib_${USER}_${SLURM_JOB_ID}}
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export PASTE_JAX_PLATFORMS=cuda
export JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p "${MPLCONFIGDIR}" "${SCRIPT_DIR}/slurm_logs"
cd "${REPO}"

echo "[gpu-scaling] host=$(hostname)"
echo "[gpu-scaling] config=${CONFIG}"
echo "[gpu-scaling] nside_list=${NSIDE_LIST}"
echo "[gpu-scaling] nhalos_list=${NHALOS_LIST}"
echo "[gpu-scaling] sample_mode=${SAMPLE_MODE}"
echo "[gpu-scaling] pixel_workers=${PIXEL_WORKERS}"
echo "[gpu-scaling] pixel_start_method=${PIXEL_START_METHOD}"
echo "[gpu-scaling] pixel_backend=${PIXEL_BACKEND}"
echo "[gpu-scaling] query_disc_buffer_safety_factor=${QUERY_DISC_BUFFER_SAFETY_FACTOR}"
echo "[gpu-scaling] include_galaxies=${INCLUDE_GALAXIES}"
echo "[gpu-scaling] galaxy_population_backend=${GALAXY_POPULATION_BACKEND}"
echo "[gpu-scaling] galaxy_compact_max_satellite_groups=${GALAXY_COMPACT_MAX_SATELLITE_GROUPS}"

for NSIDE in ${NSIDE_LIST}; do
  for NHALOS in ${NHALOS_LIST}; do
    BACKEND_TAG=${PIXEL_BACKEND//-/_}
    SAMPLE_TAG=${SAMPLE_MODE//-/_}
    HOD_TAG=${GALAXY_POPULATION_BACKEND//-/_}
    OUT="${REPO}/data/xDESI/processed/abacus_backlight/stage31_pz3_cap600/measurements/gpu_chunk_scaling_${BACKEND_TAG}_${HOD_TAG}_${SAMPLE_TAG}_nside${NSIDE}_halos${NHALOS}_job${SLURM_JOB_ID}.json"
    echo "[gpu-scaling] backend=${PIXEL_BACKEND} hod_backend=${GALAXY_POPULATION_BACKEND} sample_mode=${SAMPLE_MODE} nside=${NSIDE} n_halos=${NHALOS} output=${OUT}"
    GALAXY_ARG=--no-include-galaxies
    if [ "${INCLUDE_GALAXIES}" -gt 0 ]; then
      GALAXY_ARG=--include-galaxies
    fi
    if ! srun --ntasks=1 --nodes=1 --gres=gpu:1 --cpus-per-task="${PIXEL_WORKERS}" --cpu-bind=none \
      "${PYTHON}" "${DRIVER}" benchmark-gpu-chunk \
      --config "${CONFIG}" \
      --nside "${NSIDE}" \
      --n-halos "${NHALOS}" \
      --sample-mode "${SAMPLE_MODE}" \
      --pixel-workers "${PIXEL_WORKERS}" \
      --pixel-pool-chunksize "${PIXEL_POOL_CHUNKSIZE}" \
      --pixel-start-method "${PIXEL_START_METHOD}" \
      --pixel-backend "${PIXEL_BACKEND}" \
      --query-disc-buffer-safety-factor "${QUERY_DISC_BUFFER_SAFETY_FACTOR}" \
      --single-pixel-angle-factor 0.5 \
      "${GALAXY_ARG}" \
      --galaxy-population-backend "${GALAXY_POPULATION_BACKEND}" \
      --galaxy-compact-max-satellite-groups "${GALAXY_COMPACT_MAX_SATELLITE_GROUPS}" \
      --output "${OUT}" \
      --quiet; then
      echo "[gpu-scaling] FAILED backend=${PIXEL_BACKEND} hod_backend=${GALAXY_POPULATION_BACKEND} sample_mode=${SAMPLE_MODE} nside=${NSIDE} n_halos=${NHALOS}" >&2
    fi
  done
done
