#!/usr/bin/env bash
#SBATCH --job-name=pz3cap2400_hmc_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/abacus_paste/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
SCRIPT_DIR=${REPO}/notebooks/xDESI/abacus_paste
PIPELINE=${SCRIPT_DIR}/submit_stage31_pz3_cap2400_hmcbestfit_pipeline.sh
HMC_COMBINED=${HMC_COMBINED:-${REPO}/notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_midres2048_true_nz_hmc_stage31_multigpu/stage31_hmc_abacus_cosmo_midres2048_simple1h2h_lmax4096_gk1024_mmin11p147538_depth6_acc095_2000x16_v1/combined}
COMBINED_SUFFIX=${COMBINED_SUFFIX:-stage31_multigpu_abacus_cosmo_midres2048_simple1h2h_lmax4096_gk1024_mmin11p147538_depth6_acc095_2000x16_v1}
FIT_SUMMARY=${FIT_SUMMARY:-${HMC_COMBINED}/fit_summary_${COMBINED_SUFFIX}.json}
BESTFIT_PARAMS=${BESTFIT_PARAMS:-${HMC_COMBINED}/bestfit_params_${COMBINED_SUFFIX}.yaml}
BESTFIT_VECTOR=${BESTFIT_VECTOR:-${HMC_COMBINED}/bestfit_full_theory_data_vector_${COMBINED_SUFFIX}.npz}
REQUIRE_CONVERGENCE=${REQUIRE_CONVERGENCE:-1}
KSZ_VELOCITY_MODE=${KSZ_VELOCITY_MODE:-photoz_reconstruction_emulation}
KSZ_RECONSTRUCTION_NOISE_SEED=${KSZ_RECONSTRUCTION_NOISE_SEED:-12345}
KSZ_YLIM_MIN=${KSZ_YLIM_MIN:--5e-5}
KSZ_YLIM_MAX=${KSZ_YLIM_MAX:-5e-5}

export HMC_COMBINED COMBINED_SUFFIX FIT_SUMMARY BESTFIT_PARAMS BESTFIT_VECTOR REQUIRE_CONVERGENCE
export KSZ_VELOCITY_MODE KSZ_RECONSTRUCTION_NOISE_SEED KSZ_YLIM_MIN KSZ_YLIM_MAX
export REQUIRE_BESTFIT=1

mkdir -p "${SCRIPT_DIR}/slurm_logs"
cd "${REPO}"

echo "[cap2400-gate] host=$(hostname)"
echo "[cap2400-gate] fit_summary=${FIT_SUMMARY}"
echo "[cap2400-gate] bestfit_params=${BESTFIT_PARAMS}"
echo "[cap2400-gate] bestfit_vector=${BESTFIT_VECTOR}"
echo "[cap2400-gate] require_convergence=${REQUIRE_CONVERGENCE}"
echo "[cap2400-gate] ksz_velocity_mode=${KSZ_VELOCITY_MODE}"
echo "[cap2400-gate] ksz_ylim=${KSZ_YLIM_MIN} ${KSZ_YLIM_MAX}"

exec bash "${PIPELINE}"
