#!/bin/bash
# Submission wrapper: Stage31 64-param HMC, v2 prior, 8000 samples/chain.
#
#   * prior: priors_..._64param_desy3_fcen_v2.yaml  (referenced by the CONFIG below)
#   * sparse-l=192 (nell_compute=192 in the config)  -> 4 vec chains/GPU fit 80 GB
#   * 4 vectorized chains/GPU x 4 H100 = 16 chains
#   * each chain seeded from an EXPLICIT point (no ball): point 0 = 64-param MAP,
#     points 1..15 = 15 lowest-chi2 distinct samples from the previous sparse-192 run
#     (file below; worker r uses rows [4r:4r+4]).
#   * 100 warmup, 8000 samples/chain, checkpoint every 100 samples/chain.
#   * after each checkpoint: combine all 16 chains, best-fit + D_ell plots, AND
#     GetDist triangles for ALL 64 parameters (+ the gas/HOD/fcen/IA subsets).
#
# Usage:  bash notebooks/xDESI/survey_measure/submit_hmc_stage31_sparse192_64param_priorv2_8000.sh
set -euo pipefail

REPO_ROOT="/mnt/ceph/users/spandey/ltu-godmax/GODMAX"
SURVEY="${REPO_ROOT}/notebooks/xDESI/survey_measure"

# config that references the v2 prior + num_samples=8000 + nell_compute=192
export CONFIG="${REPO_ROOT}/param_files/xDESI/params_multiprobe_midres2048_hmc_stage31_abacus_cosmo_simple1h2h_lmax3000_gk1000_depth6_defaultacc_warm100_8000_64param_fcen_relaxedmin_lmax2000_priorv2.yaml"

# explicit per-chain init: 16 points (point0=MAP, 1..15=15 best from previous run)
export INIT_CHAIN_VALUES_FILE="${REPO_ROOT}/param_files/xDESI/init_chain_values_stage31_64param_priorv2_15best_plus_map.yaml"

# all-64 GetDist (also keeps the gas/HOD/fcen/IA subset triangles) at each checkpoint
export GETDIST_SCRIPT="${SURVEY}/plot_stage31_getdist_all64_checkpoint.py"

# checkpoint + visualization cadence (sampler counts come from CONFIG.sampler:
# num_warmup=100, num_samples=8000, num_chains=4, vectorized, max_tree_depth=6)
export CHECKPOINT_SAMPLES_EVERY="100"
export CHECKPOINT_COMBINE_AFTER="1"
export CHECKPOINT_GETDIST_AFTER="1"
export CHECKPOINT_PASTE_AFTER="0"

export RUN_VERSION="abacus_cosmo_midres2048_simple1h2h_lmax3000_gk1000_mmin11p147538_64param_fcen_relaxedmin_lmax2000_priorv2_sparse192_init15best_map_warm100_8000x16_checkpoint100_depth6_v1"

exec sbatch "${SURVEY}/submit_godmax_hmc_stage31_multigpu.sh" "$@"
