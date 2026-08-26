#!/bin/bash
#SBATCH --job-name=msbi_nulls
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/stage1_nulls_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/stage1_nulls_%j.err
#
# Stage 1: prove the parameterized paste path changed nothing it should not have,
# and measure the one-theta cost before any array is submitted.
#
#   Null A  paste split 0 with the gas override set to the FROZEN values and require
#           the y/tau/kappa maps to be bitwise identical to the archived split.
#           Proves the override machinery is inert when it should be.
#   Null B  paste split 0 with get_galmap: false and require the same bitwise
#           equality.  This is what licenses skipping 66% of the chunk loop
#           (4.20 -> ~1.85 GPU-h per point).  If it FAILS, regenerate the paste plan
#           with --keep-galaxy-map and re-cost the campaign at 4.20 GPU-h.
#   bench   the per-split timing json is the one-theta benchmark: compile time,
#           steady-state time, peak memory and output size.
#
# Null C (split-count invariance) needs a full 2 x 32 paste plus two combines and is
# therefore a separate job; it is NOT required before Stage 2, only before Stage 3.

set -uo pipefail
REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
OUT=$REPO/data/SBI_validate/mock_sbi
NULLDIR=$OUT/stage1_nulls
export JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=/tmp/matplotlib
cd "$REPO"

MAPDIR=$REPO/data/SBI_validate/three_probe_mock/maps
REFSPLIT=$MAPDIR/c0000_z0p3_0p5_mmin5e11_cosh32_fast/abacus_pasted_maps_c0000_z0p3_0p5_mmin5e11_nside1024_split000of032.h5
CAT=c0000_z0p3_0p5_mmin5e11

echo "### generating the two null configs at the frozen parameter point"
cd "$REPO/notebooks/SBI_validate"
$PY mock_sbi_design.py --round flatness --count 0 --include-reference-point \
    --keep-galaxy-map --run-name-suffix nullA_galon  --output-dir "$NULLDIR"
$PY mock_sbi_design.py --round flatness --count 0 --include-reference-point \
    --run-name-suffix nullB_galoff --output-dir "$NULLDIR"
cd "$REPO"

status=0
for tag in nullA_galon nullB_galoff; do
  CFG=$(ls "$NULLDIR"/round_flatness/*"$tag".yaml)
  RUN=$(basename "$CFG" .yaml)
  echo "############ paste $tag  $(date -Is)"
  # --overwrite so a re-run is idempotent.  Without it the paste does the full 8
  # minutes of work and then refuses at the write, which reads as a failed null when
  # the null itself passed.
  $PY notebooks/xDESI/paste_abacus_maps.py --config "$CFG" --catalog "$CAT" \
      --split-index 0 --num-splits 32 --nside 1024 --overwrite || status=1
  TRIAL=$MAPDIR/$RUN/abacus_pasted_maps_${CAT}_nside1024_split000of032.h5
  echo "############ compare $tag against the archived split"
  $PY notebooks/SBI_validate/06_paste_vs_theory/validate_mock_sbi_paste_nulls.py \
      --a "$REFSPLIT" --b "$TRIAL" --label "$tag" --require bitwise \
      --output "$NULLDIR/${tag}_report.json" || status=1
  echo "############ one-theta benchmark for $tag"
  $PY -c "
import json,sys
d=json.load(open('$TRIAL.timing.json'))
keys=('total_time_s','jax_module_import_time_s','hdf5_read_time_s','pre_chunk_setup_time_s',
      'setup_sim_map_main_time_s','profiles_time_s','base_class_time_s','write_h5_time_s',
      'n_halos','n_galaxies','chunk_size')
for k in keys: print(f'  {k:34s} {d.get(k)}')
import os; print(f'  {\"output_bytes\":34s} {os.path.getsize(\"$TRIAL\")}')
print(f'  projected GPU-hours per theta (32 splits): {32*d[\"total_time_s\"]/3600:.2f}')
"
done
exit $status
