#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
SCRIPT_DIR=${REPO}/notebooks/xDESI/abacus_paste
RUN_ROOT=${RUN_ROOT:-${REPO}/data/xDESI/processed/abacus_backlight/stage31_allpz_cap2400_hmcfailed_mmin11p147538}
PYTHON=${PYTHON:-/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python}

NSIDE=${NSIDE:-2048}
LMAX=${LMAX:-4096}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-180}
GPU_SAMPLE_INTERVAL=${GPU_SAMPLE_INTERVAL:-10}
PIXEL_WORKERS=${PIXEL_WORKERS:-16}
ALLZ_NUM_SPLITS=${ALLZ_NUM_SPLITS:-16}
PZ_NUM_SPLITS=${PZ_NUM_SPLITS:-4}
ALLZ_PASTE_ARRAY_MAX_CONCURRENT=${ALLZ_PASTE_ARRAY_MAX_CONCURRENT:-4}
PZ_PASTE_ARRAY_MAX_CONCURRENT=${PZ_PASTE_ARRAY_MAX_CONCURRENT:-1}
OVERWRITE_CATALOG=${OVERWRITE_CATALOG:-0}

DO_GENERATE_CONFIGS=${DO_GENERATE_CONFIGS:-1}
DO_SUBMIT_PASTE=${DO_SUBMIT_PASTE:-1}
DO_SUBMIT_MEASURE=${DO_SUBMIT_MEASURE:-1}

CONFIG_ALLZ=${CONFIG_ALLZ:-${SCRIPT_DIR}/stage31_allz_cap2400_hmcfailed_mmin11p147538_nside2048_lmax4096.selected.yaml}
CONFIG_PZ1=${CONFIG_PZ1:-${SCRIPT_DIR}/stage31_pz1_cap2400_hmcfailed_mmin11p147538_nside2048_lmax4096.selected.yaml}
CONFIG_PZ2=${CONFIG_PZ2:-${SCRIPT_DIR}/stage31_pz2_cap2400_hmcfailed_mmin11p147538_nside2048_lmax4096.selected.yaml}
CONFIG_PZ3=${CONFIG_PZ3:-${SCRIPT_DIR}/stage31_pz3_cap2400_hmcfailed_mmin11p147538_nside2048_lmax4096.selected.yaml}
CONFIG_PZ4=${CONFIG_PZ4:-${SCRIPT_DIR}/stage31_pz4_cap2400_hmcfailed_mmin11p147538_nside2048_lmax4096.selected.yaml}

CAT_ALLZ=${RUN_ROOT}/halos/abacus_c9999_ph9999_allzcap2400_hmcfailed_z0p001_1p2_logMgt11p147538_halos.h5
CAT_PZ1=${RUN_ROOT}/halos/abacus_c9999_ph9999_pz1cap2400_hmcfailed_z0p30_0p62_logMgt11p147538_halos.h5
CAT_PZ2=${RUN_ROOT}/halos/abacus_c9999_ph9999_pz2cap2400_hmcfailed_z0p431_0p804_logMgt11p147538_halos.h5
CAT_PZ4=${RUN_ROOT}/halos/abacus_c9999_ph9999_pz4cap2400_hmcfailed_z0p713_1p19_logMgt11p147538_halos.h5

MAP_ALLZ=${RUN_ROOT}/maps/stage31_allz_cap2400_hmcfailed_mmin11p147538_nside${NSIDE}_lmax${LMAX}/abacus_pasted_maps_allzcap2400_hmcfailed_z0p001_1p2_logMgt11p147538_nside${NSIDE}.h5
MAP_PZ1=${RUN_ROOT}/maps/stage31_pz1_cap2400_hmcfailed_mmin11p147538_nside${NSIDE}_lmax${LMAX}/abacus_pasted_maps_pz1cap2400_hmcfailed_z0p30_0p62_logMgt11p147538_nside${NSIDE}.h5
MAP_PZ2=${RUN_ROOT}/maps/stage31_pz2_cap2400_hmcfailed_mmin11p147538_nside${NSIDE}_lmax${LMAX}/abacus_pasted_maps_pz2cap2400_hmcfailed_z0p431_0p804_logMgt11p147538_nside${NSIDE}.h5
MAP_PZ4=${RUN_ROOT}/maps/stage31_pz4_cap2400_hmcfailed_mmin11p147538_nside${NSIDE}_lmax${LMAX}/abacus_pasted_maps_pz4cap2400_hmcfailed_z0p713_1p19_logMgt11p147538_nside${NSIDE}.h5
MAP_PZ3=${MAP_PZ3:-${REPO}/data/xDESI/processed/abacus_backlight/stage31_pz3_cap2400_hmcfailed_mmin11p147538/maps/stage31_pz3_cap2400_hmcfailed_mmin11p147538_nside${NSIDE}_lmax${LMAX}/abacus_pasted_maps_pz3cap2400_hmcfailed_z0p63_0p98_logMgt11p147538_nside${NSIDE}.h5}

mkdir -p "${SCRIPT_DIR}/slurm_logs" "${RUN_ROOT}/measurements" "${RUN_ROOT}/plots"
cd "${REPO}"

if [ "${DO_GENERATE_CONFIGS}" -gt 0 ]; then
  "${PYTHON}" "${SCRIPT_DIR}/make_stage31_allpz_cap2400_hmcfailed_configs.py"
fi

if [ ! -f "${MAP_PZ3}" ]; then
  echo "Missing reused pz3 map: ${MAP_PZ3}" >&2
  exit 2
fi

MEASURE_DEPS=()

submit_paste_product() {
  local label=$1
  local config=$2
  local catalog=$3
  local final_map=$4
  local num_splits=$5
  local max_concurrent=$6
  local dependency=()

  echo "[allpz-pipeline] ${label}: config=${config}"
  if [ "${DO_SUBMIT_PASTE}" -le 0 ]; then
    echo "[allpz-pipeline] ${label}: paste submission disabled"
    return
  fi

  if [ -f "${catalog}" ]; then
    echo "[allpz-pipeline] ${label}: reusing catalog ${catalog}"
  else
    local preprocess_job
    preprocess_job=$(sbatch --parsable \
      --export=ALL,CONFIG="${config}",MONITOR_INTERVAL="${MONITOR_INTERVAL}",GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL}",OVERWRITE_CATALOG="${OVERWRITE_CATALOG}" \
      "${SCRIPT_DIR}/submit_stage31_pz3_cap600_preprocess.sbatch")
    dependency=(--dependency=afterok:"${preprocess_job}")
    echo "[allpz-pipeline] ${label}: submitted preprocess ${preprocess_job}"
  fi

  if [ -f "${final_map}" ]; then
    echo "[allpz-pipeline] ${label}: reusing final map ${final_map}"
  else
    local paste_job
    local combine_job
    paste_job=$(sbatch --parsable \
      "${dependency[@]}" \
      --array=0-$((num_splits - 1))%"${max_concurrent}" \
      --export=ALL,CONFIG="${config}",NSIDE="${NSIDE}",NUM_SPLITS="${num_splits}",PIXEL_WORKERS="${PIXEL_WORKERS}",MONITOR_INTERVAL="${MONITOR_INTERVAL}",GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL}" \
      "${SCRIPT_DIR}/submit_stage31_pz3_cap600_paste_array.sbatch")
    combine_job=$(sbatch --parsable \
      --dependency=afterok:"${paste_job}" \
      --export=ALL,CONFIG="${config}",NSIDE="${NSIDE}",NUM_SPLITS="${num_splits}" \
      "${SCRIPT_DIR}/submit_stage31_allpz_cap2400_combine_only.sbatch")
    MEASURE_DEPS+=("${combine_job}")
    echo "[allpz-pipeline] ${label}: submitted paste ${paste_job}"
    echo "[allpz-pipeline] ${label}: submitted combine ${combine_job}"
  fi
}

submit_paste_product "allz-continuous" "${CONFIG_ALLZ}" "${CAT_ALLZ}" "${MAP_ALLZ}" "${ALLZ_NUM_SPLITS}" "${ALLZ_PASTE_ARRAY_MAX_CONCURRENT}"
submit_paste_product "pz1-galaxies" "${CONFIG_PZ1}" "${CAT_PZ1}" "${MAP_PZ1}" "${PZ_NUM_SPLITS}" "${PZ_PASTE_ARRAY_MAX_CONCURRENT}"
submit_paste_product "pz2-galaxies" "${CONFIG_PZ2}" "${CAT_PZ2}" "${MAP_PZ2}" "${PZ_NUM_SPLITS}" "${PZ_PASTE_ARRAY_MAX_CONCURRENT}"
submit_paste_product "pz4-galaxies" "${CONFIG_PZ4}" "${CAT_PZ4}" "${MAP_PZ4}" "${PZ_NUM_SPLITS}" "${PZ_PASTE_ARRAY_MAX_CONCURRENT}"

measure_dependency=()
if [ "${#MEASURE_DEPS[@]}" -gt 0 ]; then
  joined=$(IFS=:; echo "${MEASURE_DEPS[*]}")
  measure_dependency=(--dependency=afterok:"${joined}")
  echo "[allpz-pipeline] measure dependency afterok:${joined}"
fi

measure_job=""
if [ "${DO_SUBMIT_MEASURE}" -gt 0 ]; then
  measure_job=$(sbatch --parsable \
    "${measure_dependency[@]}" \
    --export=ALL,RUN_ROOT="${RUN_ROOT}",CONFIG="${CONFIG_ALLZ}",PZ1_CONFIG="${CONFIG_PZ1}",PZ2_CONFIG="${CONFIG_PZ2}",PZ3_CONFIG="${CONFIG_PZ3}",PZ4_CONFIG="${CONFIG_PZ4}",CONTINUOUS_MAPS="${MAP_ALLZ}",PZ1_MAPS="${MAP_PZ1}",PZ2_MAPS="${MAP_PZ2}",PZ3_MAPS="${MAP_PZ3}",PZ4_MAPS="${MAP_PZ4}",NSIDE="${NSIDE}",LMAX="${LMAX}" \
    "${SCRIPT_DIR}/submit_stage31_allpz_cap2400_measure_plot.sbatch")
  echo "[allpz-pipeline] submitted measure/plot ${measure_job}"
else
  echo "[allpz-pipeline] measure submission disabled"
fi

echo "[allpz-pipeline] run_root=${RUN_ROOT}"
echo "[allpz-pipeline] allz_map=${MAP_ALLZ}"
echo "[allpz-pipeline] pz1_map=${MAP_PZ1}"
echo "[allpz-pipeline] pz2_map=${MAP_PZ2}"
echo "[allpz-pipeline] pz3_map=${MAP_PZ3}"
echo "[allpz-pipeline] pz4_map=${MAP_PZ4}"
echo "[allpz-pipeline] measure_job=${measure_job}"
