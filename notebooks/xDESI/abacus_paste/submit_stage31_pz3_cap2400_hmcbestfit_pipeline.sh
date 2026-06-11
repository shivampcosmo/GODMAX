#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
SCRIPT_DIR=${REPO}/notebooks/xDESI/abacus_paste

CONFIG=${CONFIG:-${SCRIPT_DIR}/stage31_pz3_cap2400_hmcbestfit_mmin11p147538_nside2048_lmax4096.selected.yaml}
RUN_ROOT=${RUN_ROOT:-${REPO}/data/xDESI/processed/abacus_backlight/stage31_pz3_cap2400_hmcbestfit_mmin11p147538}
HMC_COMBINED=${HMC_COMBINED:-${REPO}/notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_midres2048_true_nz_hmc_stage31_multigpu/stage31_hmc_abacus_cosmo_midres2048_simple1h2h_lmax4096_gk1024_mmin11p147538_depth6_acc095_2000x16_v1/combined}
COMBINED_SUFFIX=${COMBINED_SUFFIX:-stage31_multigpu_abacus_cosmo_midres2048_simple1h2h_lmax4096_gk1024_mmin11p147538_depth6_acc095_2000x16_v1}
BESTFIT_PARAMS=${BESTFIT_PARAMS:-${HMC_COMBINED}/bestfit_params_${COMBINED_SUFFIX}.yaml}
BESTFIT_VECTOR=${BESTFIT_VECTOR:-${HMC_COMBINED}/bestfit_full_theory_data_vector_${COMBINED_SUFFIX}.npz}
FIT_SUMMARY=${FIT_SUMMARY:-${HMC_COMBINED}/fit_summary_${COMBINED_SUFFIX}.json}
PYTHON=${PYTHON:-/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python}

NSIDE=${NSIDE:-2048}
LMAX=${LMAX:-4096}
NUM_SPLITS=${NUM_SPLITS:-4}
PIXEL_WORKERS=${PIXEL_WORKERS:-16}
PASTE_ARRAY_MAX_CONCURRENT=${PASTE_ARRAY_MAX_CONCURRENT:-${NUM_SPLITS}}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-180}
GPU_SAMPLE_INTERVAL=${GPU_SAMPLE_INTERVAL:-10}
OVERWRITE_CATALOG=${OVERWRITE_CATALOG:-0}
REQUIRE_BESTFIT=${REQUIRE_BESTFIT:-1}
REQUIRE_CONVERGENCE=${REQUIRE_CONVERGENCE:-1}
DEPENDENCY_JOB_ID=${DEPENDENCY_JOB_ID:-}
WRITE_RUNTIME_CONFIG=${WRITE_RUNTIME_CONFIG:-1}

DO_PREPROCESS=${DO_PREPROCESS:-1}
DO_PASTE=${DO_PASTE:-1}
DO_COMBINE=${DO_COMBINE:-1}
DO_PASTED_THEORY=${DO_PASTED_THEORY:-1}
DO_DIRECT_FIELD=${DO_DIRECT_FIELD:-1}
DO_PLUS_DIRECT=${DO_PLUS_DIRECT:-1}

SHELL_INDEX_MOD=${SHELL_INDEX_MOD:-2}
BATCH_PARENT_PIXELS=${BATCH_PARENT_PIXELS:-262144}
SIM_MATCHED_TRANSFERS=${SIM_MATCHED_TRANSFERS:-1}
KSZ_VELOCITY_MODE=${KSZ_VELOCITY_MODE:-photoz_reconstruction_emulation}
KSZ_RECONSTRUCTION_NOISE_SEED=${KSZ_RECONSTRUCTION_NOISE_SEED:-12345}
KSZ_YLIM_MIN=${KSZ_YLIM_MIN:--5e-5}
KSZ_YLIM_MAX=${KSZ_YLIM_MAX:-5e-5}
DIRECT_FIELD_PYTHON=${DIRECT_FIELD_PYTHON:-/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python}
DIRECT_FIELD_CONDA_ENV_HOOK=${DIRECT_FIELD_CONDA_ENV_HOOK:-/tmp/nonexistent_godmax_hook}

MAPS=${MAPS:-${RUN_ROOT}/maps/stage31_pz3_cap2400_hmcbestfit_mmin11p147538_nside${NSIDE}_lmax${LMAX}/abacus_pasted_maps_pz3cap2400_hmcbestfit_z0p63_0p98_logMgt11p147538_nside${NSIDE}.h5}
PLUS_MAPS=${PLUS_MAPS:-${MAPS%.h5}_plus_direct_field_shells.h5}
SIM_MEAS=${SIM_MEAS:-${RUN_ROOT}/measurements/sim_pz3_cap2400_hmcbestfit_mmin11p147538_nside${NSIDE}_lmax${LMAX}_nbin10_linear.h5}
PLUS_SIM_MEAS=${PLUS_SIM_MEAS:-${RUN_ROOT}/measurements/sim_pz3_cap2400_hmcbestfit_mmin11p147538_plus_direct_field_nside${NSIDE}_lmax${LMAX}_nbin10_linear.h5}

FULL_AREA_PLOT=${FULL_AREA_PLOT:-${RUN_ROOT}/plots/stage31_pz3_cap2400_hmcbestfit_mmin11p147538_full_area_data_bestfit_with_cap_sim_nside${NSIDE}_lmax${LMAX}_Dell.pdf}
SUM_THEORY=${SUM_THEORY:-${RUN_ROOT}/theory/stage31_pz3_cap2400_hmcbestfit_mmin11p147538_nside${NSIDE}_lmax${LMAX}_theory_poweradd_sum_for_sim_measurement_matched_transfers.h5}
RESPONSE_THEORY=${RESPONSE_THEORY:-${RUN_ROOT}/theory/stage31_pz3_cap2400_hmcbestfit_mmin11p147538_nside${NSIDE}_lmax${LMAX}_theory_response_for_sim_measurement_matched_transfers.h5}
VARIANT_PLOT=${VARIANT_PLOT:-${RUN_ROOT}/plots/stage31_pz3_cap2400_hmcbestfit_mmin11p147538_pasted_only_full_data_theory_variants_with_cap_sim_Dell.pdf}

PLUS_SUM_THEORY=${PLUS_SUM_THEORY:-${RUN_ROOT}/theory/stage31_pz3_cap2400_hmcbestfit_mmin11p147538_plus_direct_field_nside${NSIDE}_lmax${LMAX}_theory_poweradd_sum_for_sim_measurement_matched_transfers.h5}
PLUS_RESPONSE_THEORY=${PLUS_RESPONSE_THEORY:-${RUN_ROOT}/theory/stage31_pz3_cap2400_hmcbestfit_mmin11p147538_plus_direct_field_nside${NSIDE}_lmax${LMAX}_theory_response_for_sim_measurement_matched_transfers.h5}
PLUS_VARIANT_PLOT=${PLUS_VARIANT_PLOT:-${RUN_ROOT}/plots/stage31_pz3_cap2400_hmcbestfit_mmin11p147538_plus_direct_field_full_data_theory_variants_with_cap_sim_Dell.pdf}
RUNTIME_CONFIG=${RUNTIME_CONFIG:-${RUN_ROOT}/configs/stage31_pz3_cap2400_hmcbestfit_mmin11p147538_nside${NSIDE}_lmax${LMAX}_${COMBINED_SUFFIX}.selected.yaml}

mkdir -p "${SCRIPT_DIR}/slurm_logs" "${RUN_ROOT}/measurements" "${RUN_ROOT}/theory" "${RUN_ROOT}/plots" "${RUN_ROOT}/configs"

if [ "${REQUIRE_BESTFIT}" -gt 0 ]; then
  if [ ! -f "${BESTFIT_PARAMS}" ]; then
    echo "Missing BESTFIT_PARAMS=${BESTFIT_PARAMS}" >&2
    exit 2
  fi
  if [ ! -f "${BESTFIT_VECTOR}" ]; then
    echo "Missing BESTFIT_VECTOR=${BESTFIT_VECTOR}" >&2
    exit 2
  fi
  if [ "${REQUIRE_CONVERGENCE}" -gt 0 ]; then
    if [ ! -f "${FIT_SUMMARY}" ]; then
      echo "Missing FIT_SUMMARY=${FIT_SUMMARY}" >&2
      exit 2
    fi
    FIT_SUMMARY="${FIT_SUMMARY}" "${PYTHON}" -c '
import json
import os
import sys

path = os.environ["FIT_SUMMARY"]
with open(path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)
conv = payload.get("convergence") or payload.get("convergence_diagnostics", {})
if conv.get("passes_basic_gate") is not True:
    print(json.dumps({"fit_summary": path, "convergence": conv}, indent=2, sort_keys=True), file=sys.stderr)
    raise SystemExit("HMC convergence gate failed; not submitting cap2400 paste pipeline.")
print(json.dumps({"fit_summary": path, "convergence": conv}, indent=2, sort_keys=True))
'
  fi
fi

CONFIG_TEMPLATE="${CONFIG}"
if [ "${WRITE_RUNTIME_CONFIG}" -gt 0 ]; then
  CONFIG_TEMPLATE="${CONFIG_TEMPLATE}" \
  RUNTIME_CONFIG="${RUNTIME_CONFIG}" \
  BESTFIT_PARAMS="${BESTFIT_PARAMS}" \
  RUN_ROOT="${RUN_ROOT}" \
  COMBINED_SUFFIX="${COMBINED_SUFFIX}" \
  KSZ_VELOCITY_MODE="${KSZ_VELOCITY_MODE}" \
  KSZ_RECONSTRUCTION_NOISE_SEED="${KSZ_RECONSTRUCTION_NOISE_SEED}" \
  "${PYTHON}" -c '
import os
from pathlib import Path
import yaml

template = Path(os.environ["CONFIG_TEMPLATE"]).expanduser().resolve()
runtime = Path(os.environ["RUNTIME_CONFIG"]).expanduser().resolve()
bestfit = Path(os.environ["BESTFIT_PARAMS"]).expanduser().resolve()
run_root = Path(os.environ["RUN_ROOT"]).expanduser().resolve()

with open(template, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

cfg.setdefault("project", {})["output_root"] = str(run_root)
godmax = cfg.setdefault("godmax", {})
godmax["bestfit_params"] = str(bestfit)
godmax["bestfit_params_source"] = "validated_hmc_convergence_gate"
godmax["hmc_combined_suffix"] = os.environ["COMBINED_SUFFIX"]
pasting = cfg.setdefault("pasting", {})
pasting["ksz_velocity_mode"] = os.environ["KSZ_VELOCITY_MODE"]
pasting["ksz_reconstruction_noise_seed"] = int(os.environ["KSZ_RECONSTRUCTION_NOISE_SEED"])

runtime.parent.mkdir(parents=True, exist_ok=True)
with open(runtime, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
print(runtime)
'
  CONFIG="${RUNTIME_CONFIG}"
  echo "Runtime config: ${CONFIG}"
  echo "Runtime config template: ${CONFIG_TEMPLATE}"
fi

dependency=()
if [ -n "${DEPENDENCY_JOB_ID}" ]; then
  dependency=(--dependency=afterok:"${DEPENDENCY_JOB_ID}")
  echo "Using initial dependency afterok:${DEPENDENCY_JOB_ID}"
fi

if [ "${DO_PREPROCESS}" -gt 0 ]; then
  preprocess_job=$(sbatch --parsable \
    "${dependency[@]}" \
    --export=ALL,CONFIG="${CONFIG}",MONITOR_INTERVAL="${MONITOR_INTERVAL}",GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL}",OVERWRITE_CATALOG="${OVERWRITE_CATALOG}" \
    "${SCRIPT_DIR}/submit_stage31_pz3_cap600_preprocess.sbatch")
  dependency=(--dependency=afterok:"${preprocess_job}")
  echo "Submitted preprocess job: ${preprocess_job}"
else
  echo "Skipping preprocess; assuming ${CONFIG} catalog already exists."
fi

paste_job=""
if [ "${DO_PASTE}" -gt 0 ]; then
  paste_job=$(sbatch --parsable \
    "${dependency[@]}" \
    --array=0-$((NUM_SPLITS - 1))%"${PASTE_ARRAY_MAX_CONCURRENT}" \
    --export=ALL,CONFIG="${CONFIG}",NSIDE="${NSIDE}",NUM_SPLITS="${NUM_SPLITS}",PIXEL_WORKERS="${PIXEL_WORKERS}",MONITOR_INTERVAL="${MONITOR_INTERVAL}",GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL}" \
    "${SCRIPT_DIR}/submit_stage31_pz3_cap600_paste_array.sbatch")
  dependency=(--dependency=afterok:"${paste_job}")
  echo "Submitted paste array job: ${paste_job}"
fi

combine_job=""
if [ "${DO_COMBINE}" -gt 0 ]; then
  combine_job=$(sbatch --parsable \
    "${dependency[@]}" \
    --export=ALL,CONFIG="${CONFIG}",NSIDE="${NSIDE}",NUM_SPLITS="${NUM_SPLITS}",LMAX="${LMAX}",MONITOR_INTERVAL="${MONITOR_INTERVAL}",GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL}",SIM_MEAS="${SIM_MEAS}",FULL_AREA_PLOT="${FULL_AREA_PLOT}",BESTFIT_VECTOR="${BESTFIT_VECTOR}",FIDUCIAL_VECTOR="${BESTFIT_VECTOR}",DO_FULL_DATA_PLOT=1,KSZ_YLIM_MIN="${KSZ_YLIM_MIN}",KSZ_YLIM_MAX="${KSZ_YLIM_MAX}" \
    "${SCRIPT_DIR}/submit_stage31_pz3_cap600_combine.sbatch")
  dependency=(--dependency=afterok:"${combine_job}")
  echo "Submitted combine/measure/pasted-only data plot job: ${combine_job}"
fi

pasted_theory_job=""
if [ "${DO_PASTED_THEORY}" -gt 0 ]; then
  pasted_theory_job=$(sbatch --parsable \
    "${dependency[@]}" \
    --export=ALL,CONFIG="${CONFIG}",RUN_ROOT="${RUN_ROOT}",NSIDE="${NSIDE}",LMAX="${LMAX}",SIM_MEAS="${SIM_MEAS}",SUM_THEORY="${SUM_THEORY}",RESPONSE_THEORY="${RESPONSE_THEORY}",VARIANT_PLOT="${VARIANT_PLOT}",SIM_MATCHED_TRANSFERS="${SIM_MATCHED_TRANSFERS}",KSZ_YLIM_MIN="${KSZ_YLIM_MIN}",KSZ_YLIM_MAX="${KSZ_YLIM_MAX}" \
    "${SCRIPT_DIR}/submit_stage31_pz3_cap600_theory_variants.sbatch")
  echo "Submitted pasted-only matched-transfer theory plot job: ${pasted_theory_job}"
fi

direct_finalize_job=""
if [ "${DO_DIRECT_FIELD}" -gt 0 ]; then
  direct_cache_job=$(sbatch --parsable \
    "${dependency[@]}" \
    --array=0-$((SHELL_INDEX_MOD - 1))%"${SHELL_INDEX_MOD}" \
    --export=ALL,CONFIG="${CONFIG}",NSIDE="${NSIDE}",SHELL_INDEX_MOD="${SHELL_INDEX_MOD}",BATCH_PARENT_PIXELS="${BATCH_PARENT_PIXELS}",PYTHON="${DIRECT_FIELD_PYTHON}",CONDA_ENV_HOOK="${DIRECT_FIELD_CONDA_ENV_HOOK}" \
    "${SCRIPT_DIR}/submit_stage31_pz3_direct_field_cache.sbatch")
  direct_finalize_job=$(sbatch --parsable \
    --dependency=afterok:"${direct_cache_job}" \
    --export=ALL,CONFIG="${CONFIG}",NSIDE="${NSIDE}",NUM_SPLITS="${NUM_SPLITS}",BATCH_PARENT_PIXELS="${BATCH_PARENT_PIXELS}",PYTHON="${DIRECT_FIELD_PYTHON}",CONDA_ENV_HOOK="${DIRECT_FIELD_CONDA_ENV_HOOK}" \
    "${SCRIPT_DIR}/submit_stage31_pz3_direct_field_finalize.sbatch")
  echo "Submitted direct-field cache job: ${direct_cache_job}"
  echo "Submitted direct-field finalize job: ${direct_finalize_job}"
fi

plus_direct_job=""
if [ "${DO_PLUS_DIRECT}" -gt 0 ]; then
  if [ -n "${direct_finalize_job}" ]; then
    plus_dependency=(--dependency=afterok:"${direct_finalize_job}")
  else
    plus_dependency=("${dependency[@]}")
  fi
  plus_direct_job=$(sbatch --parsable \
    "${plus_dependency[@]}" \
    --export=ALL,CONFIG="${CONFIG}",MAPS="${PLUS_MAPS}",SIM_MEAS="${PLUS_SIM_MEAS}",SUM_THEORY="${PLUS_SUM_THEORY}",RESPONSE_THEORY="${PLUS_RESPONSE_THEORY}",VARIANT_PLOT="${PLUS_VARIANT_PLOT}",NSIDE="${NSIDE}",SIM_MATCHED_TRANSFERS="${SIM_MATCHED_TRANSFERS}",KSZ_YLIM_MIN="${KSZ_YLIM_MIN}",KSZ_YLIM_MAX="${KSZ_YLIM_MAX}" \
    "${SCRIPT_DIR}/submit_stage31_pz3_plus_direct_measure_theory.sbatch")
  echo "Submitted plus-direct matched-transfer measure/theory/plot job: ${plus_direct_job}"
fi

echo "Config: ${CONFIG}"
echo "Bestfit params: ${BESTFIT_PARAMS}"
echo "Bestfit vector: ${BESTFIT_VECTOR}"
echo "Fit summary: ${FIT_SUMMARY}"
echo "kSZ velocity mode: ${KSZ_VELOCITY_MODE}"
echo "kSZ y-limit: ${KSZ_YLIM_MIN} ${KSZ_YLIM_MAX}"
echo "Pasted-only plot: ${FULL_AREA_PLOT}"
echo "Pasted-only matched-transfer plot: ${VARIANT_PLOT}"
echo "Plus-direct matched-transfer plot: ${PLUS_VARIANT_PLOT}"
