#!/bin/bash
#SBATCH -p gpu
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=15
#SBATCH --mem=800G
#SBATCH --time=24:00:00
#SBATCH --job-name=xdesi_hmc_stage31_midres_simple1h2h_retry
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%j.err

set -euo pipefail

REPO_ROOT="/mnt/ceph/users/spandey/ltu-godmax/GODMAX"
PREV_COMBINED="${REPO_ROOT}/notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_midres2048_true_nz_hmc_stage31_multigpu/stage31_hmc_abacus_cosmo_midres2048_simple1h2h_lmax4096_gk1024_mmin11p147538_1600x16_v1/combined"
PREV_SUFFIX="stage31_multigpu_abacus_cosmo_midres2048_simple1h2h_lmax4096_gk1024_mmin11p147538_1600x16_v1"

export CONFIG="${REPO_ROOT}/param_files/xDESI/params_multiprobe_midres2048_hmc_stage31_abacus_cosmo_simple1h2h_lmax4096_gk1024_depth6_acc095_1000x2000.yaml"
export RUN_VERSION="${RUN_VERSION:-abacus_cosmo_midres2048_simple1h2h_lmax4096_gk1024_mmin11p147538_depth6_acc095_2000x16_v1}"
export COMBINED_SUFFIX="${COMBINED_SUFFIX:-stage31_multigpu_abacus_cosmo_midres2048_simple1h2h_lmax4096_gk1024_mmin11p147538_depth6_acc095_2000x16_v1}"
export RUN_BASE_DIR="${RUN_BASE_DIR:-${REPO_ROOT}/notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_midres2048_true_nz_hmc_stage31_multigpu}"
export INIT_PARAMS="${INIT_PARAMS:-${PREV_COMBINED}/bestfit_params_${PREV_SUFFIX}.yaml}"
export HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-120}"

RUN_LABEL="${1:-stage31_hmc_abacus_cosmo_midres2048_simple1h2h_lmax4096_gk1024_mmin11p147538_depth6_acc095_2000x16_v1}"

bash "${REPO_ROOT}/notebooks/xDESI/survey_measure/submit_godmax_hmc_stage31_multigpu.sh" "${RUN_LABEL}"
