#!/bin/bash
#SBATCH --job-name=msbi_comb_r1
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-127%32
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/comb_r1_%A_%a.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/comb_r1_%A_%a.err
#
# Strictly combine each point's 32 pasted splits into one map product.
#
# CPU partition on purpose: combining is HDF5 read plus accumulate, with no forward model
# and no GPU work at all. Running it on 'gen' also means it cannot eat into the 32-GPU
# ceiling that job 14 is capped at, and 'gen' schedules immediately.
#
# Submit ONLY after job 14 has fully finished. Combining a point whose splits are still
# being written would silently produce a partial map: the combine is strict about covering
# every catalog row, but a split file that exists and is incomplete is a different failure.
#
# Splits are NOT deleted by default. They are ~371 GB for the round and deleting them is
# irreversible, so set PRUNE_SPLITS=1 explicitly if you want the space back after you have
# checked the combined products.

set -euo pipefail
REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
PLAN=${PLAN:-$REPO/data/SBI_validate/mock_sbi/paste_plan/round_1/paste_plan.json}
PRUNE_SPLITS=${PRUNE_SPLITS:-0}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
cd "$REPO"

read -r CFG CAT NSIDE NSPLIT OUTDIR < <($PY - "$PLAN" "$SLURM_ARRAY_TASK_ID" <<'INNER'
import json, pathlib, sys, yaml
REPO = pathlib.Path("/mnt/ceph/users/spandey/ltu-godmax/GODMAX")
plan = json.loads(pathlib.Path(sys.argv[1]).read_text())
entry = plan["entries"][int(sys.argv[2])]
cfg = pathlib.Path(entry["config_path"])
if not cfg.is_absolute():
    cfg = REPO / cfg
c = yaml.safe_load(cfg.read_text())
paste, project = c["pasting"], c["project"]
root = pathlib.Path(project["output_root"]) / project["map_subdir"] / paste["run_name"]
print(cfg, c["resolved_theory"]["catalog_key"], int(paste["nside"]),
      int(paste["num_splits"]), root)
INNER
)

# Count the splits WITHOUT `ls | wc -l`.  Under `set -o pipefail`, a non-matching glob
# makes ls exit 2, the pipeline inherits it, and `set -e` then kills the script BEFORE the
# diagnostic below can print -- which is how this job first failed: every task exited 2
# with a completely empty log, looking like a cancellation rather than a guard firing.
PRESENT=0
for SPLIT in $(seq 0 $((NSPLIT - 1))); do
    F=$(printf "%s/abacus_pasted_maps_%s_nside%d_split%03dof%03d.h5" \
        "$OUTDIR" "$CAT" "$NSIDE" "$SPLIT" "$NSPLIT")
    # Require the .timing.json SIDECAR, not the .h5 itself.  paste_abacus_maps.py writes
    # the HDF5 IN PLACE with no atomic rename, so a task killed mid-split leaves a
    # TRUNCATED .h5 that a bare `-s` test happily accepts -- and a truncated split feeds
    # straight into the combine, which is a wrong number with no error.  The sidecar is
    # written by _write_json_atomic (tmp + os.replace) AFTER the HDF5 is closed, so its
    # presence is proof the split is complete.
    [ -s "$F.timing.json" ] && PRESENT=$((PRESENT + 1))
    if [ -e "$F" ] && [ ! -s "$F.timing.json" ]; then
        echo "  WARNING: split $SPLIT has a .h5 but no timing sidecar (truncated/in-flight)"
    fi
done
if [ "$PRESENT" -ne "$NSPLIT" ]; then
    echo "FATAL: point $SLURM_ARRAY_TASK_ID (${OUTDIR##*/}) has $PRESENT of $NSPLIT splits."
    echo "       Job 14 has not finished this point. Submit job 15 with"
    echo "       --dependency=afterok:<paste job id> instead of submitting it immediately."
    exit 1
fi

echo "combining point $SLURM_ARRAY_TASK_ID  cfg $(basename "$CFG")  $NSPLIT splits  $(date +%H:%M:%S)"
$PY notebooks/xDESI/combine_abacus_map_splits.py --config "$CFG" --catalog "$CAT" \
    --num-splits "$NSPLIT" --nside "$NSIDE" --overwrite

COMBINED="$OUTDIR/abacus_pasted_maps_${CAT}_nside${NSIDE}.h5"
[ -s "$COMBINED" ] || { echo "FATAL: combine produced no $COMBINED"; exit 1; }
echo "wrote $COMBINED ($(stat -c%s "$COMBINED") bytes)  $(date +%H:%M:%S)"

if [ "$PRUNE_SPLITS" = "1" ]; then
    echo "PRUNE_SPLITS=1: removing this point's $NSPLIT split files"
    for SPLIT in $(seq 0 $((NSPLIT - 1))); do
        rm -f "$(printf "%s/abacus_pasted_maps_%s_nside%d_split%03dof%03d.h5" \
                 "$OUTDIR" "$CAT" "$NSIDE" "$SPLIT" "$NSPLIT")"
    done
fi
