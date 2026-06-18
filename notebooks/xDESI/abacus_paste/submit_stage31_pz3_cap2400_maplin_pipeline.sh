#!/usr/bin/env bash
# Full paste for the MAP bestfit (pop8_adam2400_polish8_lbfgsb160) WITH the
# linear-N_cen HOD occupation fix (src/get_sim_maps.py). Chains, via Slurm
# dependencies:
#   (1) paste array  [GPU]   -> per-split pasted maps + galaxy catalog
#   (2) combine      [CPU]   -> stitch maps, measure sim (13-log), and a
#                               data + MAP-bestfit-theory + sim full-area D_ell plot
#   (3) styled plot  [CPU]   -> data + sim-matched-theory + sim D_ell variant plot,
#                               log-x and likelihood bands grayed (matches the MAP figure)
#
# The cap halo catalog is bestfit-independent (mass + sky selection only), so the
# existing checkpoint_000550 catalog is reused via a symlink (no preprocess).
#
# NOTE: this script SUBMITS Slurm jobs. Run it yourself on a login/submit node:
#   bash notebooks/xDESI/abacus_paste/submit_stage31_pz3_cap2400_maplin_pipeline.sh
set -euo pipefail

REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
SCRIPT_DIR=${REPO}/notebooks/xDESI/abacus_paste

CONFIG=${CONFIG:-${SCRIPT_DIR}/stage31_pz3_cap2400_maplin_mmin11p147538_lmax3000_13log.selected.yaml}
RUN_ROOT=${RUN_ROOT:-${REPO}/data/xDESI/processed/abacus_backlight/stage31_pz3_cap2400_maplin_mmin11p147538_lmax3000_13log}
NSIDE=${NSIDE:-2048}
LMAX=${LMAX:-3000}
NUM_SPLITS=${NUM_SPLITS:-4}
PIXEL_WORKERS=${PIXEL_WORKERS:-16}
PASTE_ARRAY_MAX_CONCURRENT=${PASTE_ARRAY_MAX_CONCURRENT:-${NUM_SPLITS}}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-180}
GPU_SAMPLE_INTERVAL=${GPU_SAMPLE_INTERVAL:-10}
KSZ_YLIM_MIN=${KSZ_YLIM_MIN:--5e-5}
KSZ_YLIM_MAX=${KSZ_YLIM_MAX:-5e-5}
PLOT_ELL_MAX=${PLOT_ELL_MAX:-3000}
PLOT_XSCALE=${PLOT_XSCALE:-log}

MAP_RUN=${REPO}/notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_midres2048_true_nz_map_stage31_1gpu/stage31_map_abacus_cosmo_midres2048_simple1h2h_lmax3000_gk1000_mmin11p147538_60param_13log_apo1degC2_pairmean_map_pop8_adam2400_polish8_lbfgsb160_from_pop4adam300best_localjitter_v1
BESTFIT_PARAMS=${BESTFIT_PARAMS:-${MAP_RUN}/bestfit_params_stage31_map_abacus_cosmo_midres2048_simple1h2h_lmax3000_gk1000_mmin11p147538_60param_13log_apo1degC2_pairmean_map_pop8_adam2400_polish8_lbfgsb160_from_pop4adam300best_localjitter_v1.yaml}
BESTFIT_VECTOR=${BESTFIT_VECTOR:-${MAP_RUN}/bestfit_full_theory_data_vector_stage31_map_abacus_cosmo_midres2048_simple1h2h_lmax3000_gk1000_mmin11p147538_60param_13log_apo1degC2_pairmean_map_pop8_adam2400_polish8_lbfgsb160_from_pop4adam300best_localjitter_v1.npz}

# Reuse the existing (bestfit-independent) cap halo catalog -> skip preprocess.
CATALOG_NAME=abacus_c9999_ph9999_pz3cap2400_hmcbestfit_z0p63_0p98_logMgt11p147538_halos.h5
EXISTING_CATALOG=${EXISTING_CATALOG:-${REPO}/data/xDESI/processed/abacus_backlight/stage31_pz3_cap2400_lmax3000_gk1000_60param_warm100_checkpoint_000550/halos/${CATALOG_NAME}}

SIM_MEAS=${SIM_MEAS:-${RUN_ROOT}/measurements/sim_pz3_cap2400_maplin_mmin11p147538_nside${NSIDE}_lmax${LMAX}_nbin13_log.h5}
FULL_AREA_PLOT=${FULL_AREA_PLOT:-${RUN_ROOT}/plots/stage31_pz3_cap2400_maplin_mmin11p147538_lmax3000_13log_full_area_data_bestfit_with_cap_sim_Dell.pdf}
SUM_THEORY=${SUM_THEORY:-${RUN_ROOT}/theory/stage31_pz3_cap2400_maplin_mmin11p147538_lmax3000_13log_theory_poweradd_sum_for_sim_measurement_matched_transfers.h5}
RESPONSE_THEORY=${RESPONSE_THEORY:-${RUN_ROOT}/theory/stage31_pz3_cap2400_maplin_mmin11p147538_lmax3000_13log_theory_response_for_sim_measurement_matched_transfers.h5}
VARIANT_PLOT=${VARIANT_PLOT:-${RUN_ROOT}/plots/stage31_pz3_cap2400_maplin_mmin11p147538_lmax3000_13log_full_data_theory_variants_with_cap_sim_Dell.pdf}

mkdir -p "${SCRIPT_DIR}/slurm_logs" "${RUN_ROOT}/halos" "${RUN_ROOT}/maps" \
  "${RUN_ROOT}/measurements" "${RUN_ROOT}/theory" "${RUN_ROOT}/plots"

if [ ! -e "${RUN_ROOT}/halos/${CATALOG_NAME}" ]; then
  if [ ! -f "${EXISTING_CATALOG}" ]; then
    echo "ERROR: cap halo catalog not found at ${EXISTING_CATALOG}." >&2
    echo "       Either provide EXISTING_CATALOG=<path> or run the preprocess sbatch first." >&2
    exit 2
  fi
  ln -s "${EXISTING_CATALOG}" "${RUN_ROOT}/halos/${CATALOG_NAME}"
  echo "Linked cap halo catalog -> ${RUN_ROOT}/halos/${CATALOG_NAME}"
fi

for f in "${CONFIG}" "${BESTFIT_PARAMS}" "${BESTFIT_VECTOR}"; do
  [ -f "${f}" ] || { echo "ERROR: missing required file ${f}" >&2; exit 2; }
done

echo "Config:        ${CONFIG}"
echo "Bestfit params:${BESTFIT_PARAMS}"
echo "Bestfit vector:${BESTFIT_VECTOR}"
echo "Run root:      ${RUN_ROOT}"

# (1) paste array (GPU)
paste_job=$(sbatch --parsable \
  --array=0-$((NUM_SPLITS - 1))%"${PASTE_ARRAY_MAX_CONCURRENT}" \
  --export=ALL,CONFIG="${CONFIG}",NSIDE="${NSIDE}",NUM_SPLITS="${NUM_SPLITS}",PIXEL_WORKERS="${PIXEL_WORKERS}",MONITOR_INTERVAL="${MONITOR_INTERVAL}",GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL}" \
  "${SCRIPT_DIR}/submit_stage31_pz3_cap600_paste_array.sbatch")
echo "Submitted paste array job: ${paste_job}"

# (2) combine -> stitch + measure (13-log) + data/MAP-bestfit/sim full-area plot (CPU)
combine_job=$(sbatch --parsable \
  --dependency=afterok:"${paste_job}" \
  --export=ALL,CONFIG="${CONFIG}",NSIDE="${NSIDE}",NUM_SPLITS="${NUM_SPLITS}",LMAX="${LMAX}",MONITOR_INTERVAL="${MONITOR_INTERVAL}",GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL}",SIM_MEAS="${SIM_MEAS}",FULL_AREA_PLOT="${FULL_AREA_PLOT}",BESTFIT_VECTOR="${BESTFIT_VECTOR}",FIDUCIAL_VECTOR="${BESTFIT_VECTOR}",DO_FULL_DATA_PLOT=1,KSZ_YLIM_MIN="${KSZ_YLIM_MIN}",KSZ_YLIM_MAX="${KSZ_YLIM_MAX}",PLOT_ELL_MAX="${PLOT_ELL_MAX}" \
  "${SCRIPT_DIR}/submit_stage31_pz3_cap600_combine.sbatch")
echo "Submitted combine/measure/full-area-plot job: ${combine_job}"

# (3) styled data + sim-matched-theory + sim D_ell variant plot (CPU)
plot_job=$(sbatch --parsable \
  --dependency=afterok:"${combine_job}" \
  --export=ALL,CONFIG="${CONFIG}",RUN_ROOT="${RUN_ROOT}",NSIDE="${NSIDE}",LMAX="${LMAX}",SIM_MEAS="${SIM_MEAS}",SUM_THEORY="${SUM_THEORY}",RESPONSE_THEORY="${RESPONSE_THEORY}",VARIANT_PLOT="${VARIANT_PLOT}",PLOT_ELL_MAX="${PLOT_ELL_MAX}",PLOT_XSCALE="${PLOT_XSCALE}",KSZ_YLIM_MIN="${KSZ_YLIM_MIN}",KSZ_YLIM_MAX="${KSZ_YLIM_MAX}" \
  "${SCRIPT_DIR}/submit_stage31_pz3_cap2400_maplin_lmax3000_13log_plot.sbatch")
echo "Submitted styled data+theory+sim plot job: ${plot_job}"

echo
echo "Chain: paste(${paste_job}) -> combine(${combine_job}) -> plot(${plot_job})"
echo "Final data+theory+sim D_ell plots:"
echo "  full-area (data + MAP bestfit theory + cap sim): ${FULL_AREA_PLOT}"
echo "  variants  (data + sim-matched theory + cap sim): ${VARIANT_PLOT}"
