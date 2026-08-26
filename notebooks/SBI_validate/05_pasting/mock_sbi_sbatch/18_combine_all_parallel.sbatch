#!/bin/bash
#SBATCH --job-name=msbi_comb_all
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/comball_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/comball_%j.err
#
# Combine every fully-pasted round-1 point, on ONE node, in parallel.
#
# This REPLACES the 128-task array (15_combine_round1.sbatch), which cannot work on this
# cluster:  partition `gen` is OverSubscribe=EXCLUSIVE, so a task asking for 4 CPUs is
# allocated the WHOLE 96-core node, and the gen partition QOS (`inter`) caps a user at
# cpu=256, node=4.  With a JupyterHub session holding 64 CPUs and one node, that leaves
# room for exactly one combine task at a time: `%32` in the array is meaningless, 128
# points take ~2.1 h serially, and 95 of every 96 cores sit idle.  One node running
# xargs -P does the same work in ~10 min on a single allocation.
#
# Combining one point is ~60 s: read 32 x ~90 MB of split HDF5, sum three float32 maps.
# It is IO-bound and single-threaded, so JOBS is set well above the useful core count.
#
# SAFE TO RUN WHILE THE PASTE IS STILL GOING, and safe to rerun: it only touches points
# whose 32 splits ALL carry a .timing.json sidecar (written by _write_json_atomic, tmp +
# os.replace, AFTER the HDF5 is closed -- the only proof a split is complete, since the
# paste itself writes its .h5 in place with no atomic rename).
#
# --overwrite for the same reason one level up: combine_partial_maps also writes the final
# HDF5 in place, so an existing file is not proof of a complete one.  Redoing a point is
# 60 s; RECOMBINE=0 skips points already combined if you would rather have the speed.

set -euo pipefail
REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
PLAN=${PLAN:-$REPO/data/SBI_validate/mock_sbi/paste_plan/round_1/paste_plan.json}
JOBS=${JOBS:-16}
RECOMBINE=${RECOMBINE:-0}
WORK=${WORK:-$REPO/data/SBI_validate/mock_sbi/combine_all}
export OMP_NUM_THREADS=1
mkdir -p "$WORK"
cd "$REPO"

echo "### host $(hostname)  cores $(nproc)  JOBS=$JOBS  RECOMBINE=$RECOMBINE  $(date +%H:%M:%S)"

$PY - "$PLAN" "$RECOMBINE" <<'INNER' > "$WORK/to_combine.txt"
import json, pathlib, sys, yaml
REPO = pathlib.Path("/mnt/ceph/users/spandey/ltu-godmax/GODMAX")
plan = json.loads(pathlib.Path(sys.argv[1]).read_text())
recombine = sys.argv[2] not in {"0", "", "false", "False"}
ready = combined = incomplete = 0
for entry in plan["entries"]:
    cfg = pathlib.Path(entry["config_path"])
    if not cfg.is_absolute():
        cfg = REPO / cfg
    c = yaml.safe_load(cfg.read_text())
    paste, project = c["pasting"], c["project"]
    root = pathlib.Path(project["output_root"]) / project["map_subdir"] / paste["run_name"]
    cat, nside, n = c["resolved_theory"]["catalog_key"], int(paste["nside"]), int(paste["num_splits"])
    if not all((root / f"abacus_pasted_maps_{cat}_nside{nside}_split{s:03d}of{n:03d}.h5.timing.json").is_file()
               for s in range(n)):
        incomplete += 1
        continue
    ready += 1
    final = root / f"abacus_pasted_maps_{cat}_nside{nside}.h5"
    if final.is_file() and not recombine:
        combined += 1
        continue
    print(f"{cfg} {cat} {nside} {n}")
print(f"      {ready} of {len(plan['entries'])} points fully pasted; "
      f"{incomplete} still pasting; {combined} already combined and skipped",
      file=sys.stderr)
INNER

N=$(wc -l < "$WORK/to_combine.txt")
echo "### $N point(s) to combine"
[ "$N" -gt 0 ] || { echo "### nothing to do"; exit 0; }

cat > "$WORK/combine_one.sh" <<COMBEOF
#!/bin/bash
set -euo pipefail
echo "      [\$(date +%H:%M:%S)] \$(basename "\$1")"
$PY $REPO/notebooks/xDESI/combine_abacus_map_splits.py \\
    --config "\$1" --catalog "\$2" --num-splits "\$4" --nside "\$3" --overwrite
COMBEOF
chmod +x "$WORK/combine_one.sh"

xargs -a "$WORK/to_combine.txt" -P "$JOBS" -n 4 "$WORK/combine_one.sh"
echo "### done  $(date +%H:%M:%S)"
