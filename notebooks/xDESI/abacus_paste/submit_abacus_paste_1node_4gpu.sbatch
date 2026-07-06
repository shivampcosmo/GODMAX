#!/usr/bin/env bash
#SBATCH --job-name=abacus_paste_4gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=08:00:00
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
NSIDE=${NSIDE:-512}
NUM_SPLITS=${NUM_SPLITS:-4}
PIXEL_START_METHOD=${PIXEL_START_METHOD:-fork}
PIXEL_WORKERS=${PIXEL_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}
JAX_MEM_FRACTION=${JAX_MEM_FRACTION:-0.95}

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
export XLA_PYTHON_CLIENT_MEM_FRACTION="${JAX_MEM_FRACTION}"
export PASTE_JAX_PLATFORMS=cuda
export JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p "${MPLCONFIGDIR}" "${SCRIPT_DIR}/slurm_logs"
cd "${REPO}"

echo "[paste-1node-4gpu] host=$(hostname)"
echo "[paste-1node-4gpu] config=${CONFIG}"
echo "[paste-1node-4gpu] nside=${NSIDE} num_splits=${NUM_SPLITS}"
echo "[paste-1node-4gpu] ntasks=${SLURM_NTASKS:-unset} cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}"
echo "[paste-1node-4gpu] pixel_workers=${PIXEL_WORKERS}"
echo "[paste-1node-4gpu] python=${PYTHON}"
echo "[paste-1node-4gpu] LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"
echo "[paste-1node-4gpu] CUDA_VISIBLE_DEVICES at batch start=${CUDA_VISIBLE_DEVICES:-unset}"

srun \
  --ntasks="${NUM_SPLITS}" \
  --nodes=1 \
  --gpus-per-task=1 \
  --cpus-per-task="${PIXEL_WORKERS}" \
  --gpu-bind=single:1 \
  --cpu-bind=cores \
  bash -lc '
    set -euo pipefail
    if [[ -f '"${CONDA_ENV_HOOK}"' ]]; then
      source '"${CONDA_ENV_HOOK}"'
    fi
    SPLIT_INDEX=${SLURM_PROCID}
    echo "[paste-task] host=$(hostname) rank=${SLURM_PROCID} localid=${SLURM_LOCALID} cuda=${CUDA_VISIBLE_DEVICES:-unset} split=${SPLIT_INDEX}/'"${NUM_SPLITS}"'"
    echo "[paste-task] python='"${PYTHON}"' ld=${LD_LIBRARY_PATH:-unset}"
    '"${PYTHON}"' '"${DRIVER}"' paste-split \
      --config '"${CONFIG}"' \
      --nside '"${NSIDE}"' \
      --split-index "${SPLIT_INDEX}" \
      --num-splits '"${NUM_SPLITS}"' \
      --pixel-workers '"${PIXEL_WORKERS}"' \
      --pixel-start-method '"${PIXEL_START_METHOD}"' \
      --overwrite \
      --verbose
  '
