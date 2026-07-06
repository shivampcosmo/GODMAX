#!/bin/bash
# Submission wrapper for the Stage31 64-param HMC run with the sparse-l=192 fix.
#
#   4 vectorized chains/GPU x 4 H100 = 16 chains
#   sparse-l=192 (halo_params.nell_compute=192 is set in the config below) -> fits 80 GB
#   small ball (1e-4) around the 64-param MAP, 100 warmup, 2000 samples
#   checkpoint every 100 samples/chain; after each checkpoint the monitor combines
#   ALL 16 chains and makes (a) best-fit + D_ell comparison plots and
#   (b) GetDist triangle contours for the gas/HOD/fcen/IA parameter subset.
#
# Usage:   bash notebooks/xDESI/survey_measure/submit_hmc_stage31_sparse192_64param.sh
# (this calls `sbatch` on the multi-GPU launcher; nothing is submitted until you run it)
set -euo pipefail

REPO_ROOT="/mnt/ceph/users/spandey/ltu-godmax/GODMAX"
SURVEY="${REPO_ROOT}/notebooks/xDESI/survey_measure"

# --- production config: 64-param fcen relaxedmin, lmax3000; nell_compute=192 is set
#     under comparison_overrides.params.halo_params, so the halo model + C(l) are
#     computed on 192 log-l points and interpolated back to the dense integer grid. ---
export CONFIG="${REPO_ROOT}/param_files/xDESI/params_multiprobe_midres2048_hmc_stage31_abacus_cosmo_simple1h2h_lmax3000_gk1000_depth6_defaultacc_warm100_2000_64param_fcen_relaxedmin_lmax2000.yaml"

# --- start in a very small ball around the 64-param MAP best fit ---
export INIT_PARAMS="${SURVEY}/outputs/godmax_multiprobe_midres2048_true_nz_map_stage31_1gpu/stage31_map_abacus_cosmo_midres2048_simple1h2h_lmax3000_gk1000_mmin11p147538_64param_fcen_relaxedmin_lmax2000_map_pop8_adam2400_warmcos_lr1p2em3_polish8_lbfgsb240_from_prevbest8_v1/bestfit_params_stage31_map_abacus_cosmo_midres2048_simple1h2h_lmax3000_gk1000_mmin11p147538_64param_fcen_relaxedmin_lmax2000_map_pop8_adam2400_warmcos_lr1p2em3_polish8_lbfgsb240_from_prevbest8_v1.yaml"
export INIT_BALL_SCALE="1e-4"      # very small ball: 1e-4 x prior width/sigma per param
export INIT_BALL_SEED_BASE="42000" # per-worker init-ball seeds 42000..42003

# --- sampler counts come from CONFIG.sampler (num_warmup=100, num_samples=2000,
#     num_chains=4, chain_method=vectorized, max_tree_depth=6) ---

# --- checkpoint + plotting cadence ---
export CHECKPOINT_SAMPLES_EVERY="100"   # checkpoint every 100 samples per chain
export CHECKPOINT_COMBINE_AFTER="1"     # combine all 16 chains at each checkpoint
export CHECKPOINT_GETDIST_AFTER="1"     # GetDist contours (gas/HOD/fcen/IA) per checkpoint
export CHECKPOINT_PASTE_AFTER="0"       # no map-pasting jobs

export RUN_VERSION="abacus_cosmo_midres2048_simple1h2h_lmax3000_gk1000_mmin11p147538_64param_fcen_relaxedmin_lmax2000_sparse192_ball1em4_warm100_2000x16_checkpoint100_depth6_v1"

exec sbatch "${SURVEY}/submit_godmax_hmc_stage31_multigpu.sh" "$@"
