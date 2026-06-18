#!/bin/bash
#SBATCH -p gpu
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=xdesi_map_stage31_1gpu
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%j.err

set -euo pipefail

REPO_ROOT="/mnt/ceph/users/spandey/ltu-godmax/GODMAX"
PYTHON="/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python"
RUNNER="${REPO_ROOT}/notebooks/xDESI/survey_measure/godmax_multiprobe_map_stage31.py"
CONFIG="${CONFIG:-${REPO_ROOT}/param_files/xDESI/params_multiprobe_midres2048_hmc_stage31_abacus_cosmo_simple1h2h_lmax3000_gk1000_depth6_defaultacc_warm100_2000_60param.yaml}"
HMC_RUN_DIR="${HMC_RUN_DIR:-${REPO_ROOT}/notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_midres2048_true_nz_hmc_stage31_multigpu/stage31_hmc_abacus_cosmo_midres2048_simple1h2h_lmax3000_gk1000_mmin11p147538_60param_13log_apo1degC2_pairmean_warm100_2000x16_checkpoint25_v1}"
PARAMS="${PARAMS:-}"
RUN_VERSION="${RUN_VERSION:-abacus_cosmo_midres2048_simple1h2h_lmax3000_gk1000_mmin11p147538_60param_13log_apo1degC2_pairmean_map_pop4_adam40_polish2_lbfgsb120_from_latest_hmc_v2}"
RUN_LABEL="${1:-stage31_map_${RUN_VERSION}}"
SUFFIX="${SUFFIX:-stage31_map_${RUN_VERSION}}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_midres2048_true_nz_map_stage31_1gpu}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_LABEL}}"
LOG_DIR="${REPO_ROOT}/notebooks/xDESI/survey_measure/logs"

POPULATION_SIZE="${POPULATION_SIZE:-4}"
POPULATION_STEPS="${POPULATION_STEPS:-40}"
POPULATION_LR="${POPULATION_LR:-1.0e-3}"
POPULATION_LR_SCHEDULE="${POPULATION_LR_SCHEDULE:-constant}"
POPULATION_LR_MIN_FRACTION="${POPULATION_LR_MIN_FRACTION:-0.1}"
POPULATION_LR_WARMUP_STEPS="${POPULATION_LR_WARMUP_STEPS:-0}"
POPULATION_EVAL_BATCH_SIZE="${POPULATION_EVAL_BATCH_SIZE:-0}"
RESTART_CANDIDATES="${RESTART_CANDIDATES:-}"
RESTART_TOP_K="${RESTART_TOP_K:-0}"
HMC_TOP_K="${HMC_TOP_K:-8}"
POLISH_TOP_K="${POLISH_TOP_K:-2}"
N_RESTARTS="${N_RESTARTS:-2}"
PERTURB_SCALE="${PERTURB_SCALE:-0.03}"
ADAM_STEPS="${ADAM_STEPS:-0}"
ADAM_LR="${ADAM_LR:-2.0e-3}"
ADAM_LR_SCHEDULE="${ADAM_LR_SCHEDULE:-constant}"
ADAM_LR_MIN_FRACTION="${ADAM_LR_MIN_FRACTION:-0.1}"
ADAM_LR_WARMUP_STEPS="${ADAM_LR_WARMUP_STEPS:-0}"
LBFGSB_MAXITER="${LBFGSB_MAXITER:-120}"
LBFGSB_MAXFUN="${LBFGSB_MAXFUN:-180}"
LBFGSB_FTOL="${LBFGSB_FTOL:-1.0e-7}"
LBFGSB_GTOL="${LBFGSB_GTOL:-1.0e-4}"
LBFGSB_MAXLS="${LBFGSB_MAXLS:-20}"
NORMAL_BOUND_SIGMA="${NORMAL_BOUND_SIGMA:-8.0}"
UNIFORM_EPS="${UNIFORM_EPS:-1.0e-6}"
LOG_EVERY="${LOG_EVERY:-1}"
PLOT_ELL_MAX="${PLOT_ELL_MAX:-3000}"
PLOT_XSCALE="${PLOT_XSCALE:-log}"
PLOT_XLIM="${PLOT_XLIM:-100,3000}"
RESIDUAL_YLIM="${RESIDUAL_YLIM:-}"
KSZ_SCALE="${KSZ_SCALE:-1000}"
KSZ_YLIM="${KSZ_YLIM:-}"

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
echo "[$(date)] params=${PARAMS:-latest_hmc_bestfit}"
echo "[$(date)] run_label=${RUN_LABEL}"
echo "[$(date)] suffix=${SUFFIX}"
echo "[$(date)] output_dir=${OUTPUT_DIR}"
echo "[$(date)] population_size=${POPULATION_SIZE}"
echo "[$(date)] population_steps=${POPULATION_STEPS}"
echo "[$(date)] population_lr=${POPULATION_LR}"
echo "[$(date)] population_lr_schedule=${POPULATION_LR_SCHEDULE}"
echo "[$(date)] population_lr_min_fraction=${POPULATION_LR_MIN_FRACTION}"
echo "[$(date)] population_lr_warmup_steps=${POPULATION_LR_WARMUP_STEPS}"
echo "[$(date)] population_eval_batch_size=${POPULATION_EVAL_BATCH_SIZE}"
echo "[$(date)] restart_candidates=${RESTART_CANDIDATES:-none}"
echo "[$(date)] restart_top_k=${RESTART_TOP_K}"
echo "[$(date)] hmc_top_k=${HMC_TOP_K}"
echo "[$(date)] polish_top_k=${POLISH_TOP_K}"
echo "[$(date)] n_restarts=${N_RESTARTS}"
echo "[$(date)] adam_lr_schedule=${ADAM_LR_SCHEDULE}"
echo "[$(date)] lbfgsb_maxiter=${LBFGSB_MAXITER}"
echo "[$(date)] lbfgsb_maxfun=${LBFGSB_MAXFUN}"
echo "[$(date)] plot_xscale=${PLOT_XSCALE}"
echo "[$(date)] plot_xlim=${PLOT_XLIM}"
echo "[$(date)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

ARGS=(
  --config "${CONFIG}"
  --hmc-run-dir "${HMC_RUN_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --suffix "${SUFFIX}"
  --population-size "${POPULATION_SIZE}"
  --population-steps "${POPULATION_STEPS}"
  --population-lr "${POPULATION_LR}"
  --population-lr-schedule "${POPULATION_LR_SCHEDULE}"
  --population-lr-min-fraction "${POPULATION_LR_MIN_FRACTION}"
  --population-lr-warmup-steps "${POPULATION_LR_WARMUP_STEPS}"
  --population-eval-batch-size "${POPULATION_EVAL_BATCH_SIZE}"
  --restart-top-k "${RESTART_TOP_K}"
  --hmc-top-k "${HMC_TOP_K}"
  --polish-top-k "${POLISH_TOP_K}"
  --n-restarts "${N_RESTARTS}"
  --perturb-scale "${PERTURB_SCALE}"
  --adam-steps "${ADAM_STEPS}"
  --adam-lr "${ADAM_LR}"
  --adam-lr-schedule "${ADAM_LR_SCHEDULE}"
  --adam-lr-min-fraction "${ADAM_LR_MIN_FRACTION}"
  --adam-lr-warmup-steps "${ADAM_LR_WARMUP_STEPS}"
  --lbfgsb-maxiter "${LBFGSB_MAXITER}"
  --lbfgsb-maxfun "${LBFGSB_MAXFUN}"
  --lbfgsb-ftol "${LBFGSB_FTOL}"
  --lbfgsb-gtol "${LBFGSB_GTOL}"
  --lbfgsb-maxls "${LBFGSB_MAXLS}"
  --normal-bound-sigma "${NORMAL_BOUND_SIGMA}"
  --uniform-eps "${UNIFORM_EPS}"
  --log-every "${LOG_EVERY}"
  --plot-ell-max "${PLOT_ELL_MAX}"
  --plot-xscale "${PLOT_XSCALE}"
  --plot-xlim "${PLOT_XLIM}"
  --ksz-scale "${KSZ_SCALE}"
)
if [[ -n "${PARAMS}" ]]; then
  ARGS+=(--params "${PARAMS}")
fi
if [[ -n "${RESTART_CANDIDATES}" ]]; then
  ARGS+=(--restart-candidates "${RESTART_CANDIDATES}")
fi
if [[ -n "${RESIDUAL_YLIM}" ]]; then
  ARGS+=(--residual-ylim "${RESIDUAL_YLIM}")
fi
if [[ -n "${KSZ_YLIM}" ]]; then
  ARGS+=(--ksz-ylim "${KSZ_YLIM}")
fi

"${PYTHON}" -u "${RUNNER}" "${ARGS[@]}"
