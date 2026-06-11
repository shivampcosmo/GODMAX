#!/bin/bash
#SBATCH -p gpu
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=15
#SBATCH --mem=800G
#SBATCH --time=24:00:00
#SBATCH --job-name=xdesi_hmc_stage31_abacus_cosmo
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/logs/%x.%j.err

set -euo pipefail

REPO_ROOT="/mnt/ceph/users/spandey/ltu-godmax/GODMAX"

export CONFIG="${REPO_ROOT}/param_files/xDESI/params_multiprobe_fast1024_hmc_stage31_abacus_cosmo_400x16.yaml"
export RUN_VERSION="${RUN_VERSION:-abacus_cosmo_mmin11p147538_1600x16_v1}"
export COMBINED_SUFFIX="${COMBINED_SUFFIX:-stage31_multigpu_abacus_cosmo_mmin11p147538_1600x16_v1}"
export INIT_PARAMS="${INIT_PARAMS:-${REPO_ROOT}/param_files/xDESI/params_multiprobe_fast1024_true_nz_stage31_bestfit_v2.yaml}"
export HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-120}"

RUN_LABEL="${1:-stage31_hmc_abacus_cosmo_mmin11p147538_1600x16_v1}"

bash "${REPO_ROOT}/notebooks/xDESI/survey_measure/submit_godmax_hmc_stage31_multigpu.sh" "${RUN_LABEL}"
