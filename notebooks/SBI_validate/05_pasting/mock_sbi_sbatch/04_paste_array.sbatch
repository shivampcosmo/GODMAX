#!/bin/bash
#SBATCH --job-name=msbi_paste
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/paste_%A_%a.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/paste_%A_%a.err
#
# Paste one (theta, split) pair.  Shared by Stage 2 (flatness scan) and Stages 3-5
# (the production rounds): the plan manifest decides which points exist.
#
# The array index maps as  task = theta_index * NUM_SPLITS + split_index, so an array
# of 0-(N_theta*NUM_SPLITS-1) covers the whole plan.  Submit with --array and %CONC to
# cap concurrency; every task is one A100 for ~4-8 minutes.
#
# Required environment:
#   PLAN        absolute path to paste_plan.json from mock_sbi_design.py
#   NUM_SPLITS  splits per theta (must match the plan's num_splits)
#
# Example (12 theta x 32 splits = 384 tasks, 8 at a time):
#   PLAN=.../paste_plan.json NUM_SPLITS=32 sbatch --array=0-383%8 04_paste_array.sbatch

set -euo pipefail
REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
: "${PLAN:?set PLAN to the paste_plan.json}"
: "${NUM_SPLITS:=32}"
export JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=/tmp/matplotlib
cd "$REPO"

THETA_INDEX=$(( SLURM_ARRAY_TASK_ID / NUM_SPLITS ))
SPLIT_INDEX=$(( SLURM_ARRAY_TASK_ID % NUM_SPLITS ))

read -r CFG PLAN_SPLITS CACHED < <($PY - "$PLAN" "$THETA_INDEX" <<'INNER'
import json, pathlib, sys
plan = json.loads(pathlib.Path(sys.argv[1]).read_text())
index = int(sys.argv[2])
entries = plan["entries"]
if index >= len(entries):
    raise SystemExit(f"theta index {index} exceeds the plan's {len(entries)} entries")
entry = entries[index]
path = pathlib.Path(entry["config_path"])
if not path.is_absolute():
    path = pathlib.Path("/mnt/ceph/users/spandey/ltu-godmax/GODMAX") / path
print(path, plan["num_splits"], "cached" if entry.get("cached_map") else "paste")
INNER
)

if [ "$CACHED" = "cached" ]; then
  # A forced anchor whose map already exists (the frozen paste is one of the design
  # points).  Re-pasting it would spend a GPU-hour reproducing a validated product.
  echo "theta $THETA_INDEX declares a cached map; nothing to paste for this task"
  exit 0
fi
if [ "$PLAN_SPLITS" != "$NUM_SPLITS" ]; then
  echo "NUM_SPLITS=$NUM_SPLITS does not match the plan's num_splits=$PLAN_SPLITS" >&2
  exit 1
fi

CAT=$($PY -c "
import yaml,sys; print(yaml.safe_load(open('$CFG'))['resolved_theory']['catalog_key'])")
NSIDE=$($PY -c "
import yaml,sys; print(int(yaml.safe_load(open('$CFG'))['pasting']['nside']))")

echo "task $SLURM_ARRAY_TASK_ID -> theta $THETA_INDEX split $SPLIT_INDEX  cfg $(basename "$CFG")"
$PY notebooks/xDESI/paste_abacus_maps.py --config "$CFG" --catalog "$CAT" \
    --split-index "$SPLIT_INDEX" --num-splits "$NUM_SPLITS" --nside "$NSIDE"
