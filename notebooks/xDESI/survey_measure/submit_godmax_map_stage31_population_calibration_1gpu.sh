#!/bin/bash
#SBATCH -p gpu
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=xdesi_map_popcal_1gpu
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%j.err

set -euo pipefail

REPO_ROOT="/mnt/ceph/users/spandey/ltu-godmax/GODMAX"
PYTHON="/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python"
RUNNER="${REPO_ROOT}/notebooks/xDESI/survey_measure/godmax_multiprobe_map_stage31.py"
CONFIG="${CONFIG:-${REPO_ROOT}/param_files/xDESI/params_multiprobe_midres2048_hmc_stage31_abacus_cosmo_simple1h2h_lmax3000_gk1000_depth6_defaultacc_warm100_2000_60param.yaml}"
HMC_RUN_DIR="${HMC_RUN_DIR:-${REPO_ROOT}/notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_midres2048_true_nz_hmc_stage31_multigpu/stage31_hmc_abacus_cosmo_midres2048_simple1h2h_lmax3000_gk1000_mmin11p147538_60param_13log_apo1degC2_pairmean_warm100_2000x16_checkpoint25_v1}"
RUN_VERSION="${RUN_VERSION:-abacus_cosmo_midres2048_simple1h2h_lmax3000_gk1000_mmin11p147538_60param_13log_apo1degC2_pairmean_map_population_calibration_v1}"
RUN_LABEL="${1:-stage31_map_${RUN_VERSION}}"
SUFFIX="${SUFFIX:-stage31_map_${RUN_VERSION}}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_midres2048_true_nz_map_stage31_1gpu}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_LABEL}}"
LOG_DIR="${REPO_ROOT}/notebooks/xDESI/survey_measure/logs"

BENCHMARK_POPULATION_SIZES="${BENCHMARK_POPULATION_SIZES:-1,2,4,6,8,12,16}"
BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-1}"
HMC_TOP_K="${HMC_TOP_K:-16}"
PERTURB_SCALE="${PERTURB_SCALE:-0.03}"
NORMAL_BOUND_SIGMA="${NORMAL_BOUND_SIGMA:-8.0}"
UNIFORM_EPS="${UNIFORM_EPS:-1.0e-6}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${OUTPUT_DIR}/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

if [[ -n "${CRAY_CUDATOOLKIT_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${CRAY_CUDATOOLKIT_DIR}/lib64:${LD_LIBRARY_PATH:-}"
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_cuda_data_dir=${CRAY_CUDATOOLKIT_DIR}"
fi

echo "[$(date)] host=$(hostname)"
echo "[$(date)] config=${CONFIG}"
echo "[$(date)] hmc_run_dir=${HMC_RUN_DIR}"
echo "[$(date)] run_label=${RUN_LABEL}"
echo "[$(date)] suffix=${SUFFIX}"
echo "[$(date)] output_dir=${OUTPUT_DIR}"
echo "[$(date)] benchmark_population_sizes=${BENCHMARK_POPULATION_SIZES}"
echo "[$(date)] benchmark_repeats=${BENCHMARK_REPEATS}"
echo "[$(date)] hmc_top_k=${HMC_TOP_K}"
echo "[$(date)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

"${PYTHON}" -u "${RUNNER}" \
  --config "${CONFIG}" \
  --hmc-run-dir "${HMC_RUN_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --suffix "${SUFFIX}" \
  --hmc-top-k "${HMC_TOP_K}" \
  --perturb-scale "${PERTURB_SCALE}" \
  --normal-bound-sigma "${NORMAL_BOUND_SIGMA}" \
  --uniform-eps "${UNIFORM_EPS}" \
  --benchmark-only \
  --benchmark-population-sizes "${BENCHMARK_POPULATION_SIZES}" \
  --benchmark-repeats "${BENCHMARK_REPEATS}"
