#!/bin/bash
#SBATCH --job-name=msbi_trial
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/trial_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/trial_%j.err
#
# TRIAL mock-SBI posterior on however many round-1 points have finished pasting so far.
# Safe to submit at ANY time, including while job 14 is still running, and safe to rerun.
#
# Why a partial design is statistically clean, not a shortcut. The design was drawn IID
# from a fixed mixture, so ANY subset selected by a mechanism independent of theta is still
# an IID sample from that mixture -- a smaller design, not a biased one. The selection
# mechanism here is "which points finished first", and paste cost was verified
# theta-independent (all 24 concurrent tasks reported identical pixel-neighbour pair counts
# at the same chunk index). So this is a valid design of whatever size is ready. It is
# NOISIER than the full 128, and the oracle's ladder says how much: pooled drift ran
# 0.146 / 0.137 / 0.102 / 0.050 sigma at 128 / 256 / 384 / 512 points, so BELOW 128 points
# expect worse than 0.15 sigma. Treat the output as a smoke test and an early look, not a
# result.
#
# CPU only. No GPU is requested anywhere, so this cannot touch the 24-GPU ceiling that
# job 14 is running against. It needs no forward model: the score operator and the
# observation are both read from cached artifacts.
#
# A point is READY only when all 32 of its splits have a .timing.json sidecar. The paste
# writes its HDF5 in place with no atomic rename, so a bare existence test would accept a
# split that is still being written by a live task; the sidecar is written atomically after
# the HDF5 is closed, so it is the only safe readiness marker.

set -euo pipefail
REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
PLAN=${PLAN:-$REPO/data/SBI_validate/mock_sbi/paste_plan/round_1/paste_plan.json}
WORK=${WORK:-$REPO/data/SBI_validate/mock_sbi/trial}
MIN_POINTS=${MIN_POINTS:-16}
# Points whose 32 splits carry TWO different pasting_helper_sha256 stamps.  They are fully
# pasted but combine_partial_maps refuses them, and repairing them needs a GPU repaste
# (19_repaste_helper_sha.sbatch).  Excluding them does NOT bias the design: the exclusion
# is by array index and wall clock -- which points happened to be mid-flight when the
# helper was edited -- and plan index is uncorrelated with theta (max |r| = 0.122, SE 0.088
# over 128 points), so the survivors are still an IID draw from the design mixture.
EXCLUDE=${EXCLUDE:-$REPO/data/SBI_validate/mock_sbi/paste_plan/round_1/helper_sha_repaste.json}
REPLICAS=${REPLICAS:-64}
SEEDS=${SEEDS:-4}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd "$REPO"
mkdir -p "$WORK"
TAG="n$(date +%H%M)"

echo "### step 1: which points are READY (all 32 splits, each with its timing sidecar)?"
$PY - "$PLAN" "$WORK/ready_plan.json" "$MIN_POINTS" "$EXCLUDE" <<'INNER'
import json, pathlib, sys, yaml
REPO = pathlib.Path("/mnt/ceph/users/spandey/ltu-godmax/GODMAX")
plan = json.loads(pathlib.Path(sys.argv[1]).read_text())
excluded = set()
exclude_path = pathlib.Path(sys.argv[4]) if len(sys.argv) > 4 else None
if exclude_path is not None and exclude_path.is_file():
    manifest = json.loads(exclude_path.read_text())
    excluded = {str(pathlib.Path(e["config_path"]).name) for e in manifest["entries"]}
    print(f"      excluding {len(excluded)} point(s) with mixed pasting_helper_sha256")
ready = []
for entry in plan["entries"]:
    if pathlib.Path(entry["config_path"]).name in excluded:
        continue
    cfg = pathlib.Path(entry["config_path"])
    if not cfg.is_absolute():
        cfg = REPO / cfg
    c = yaml.safe_load(cfg.read_text())
    paste, project = c["pasting"], c["project"]
    root = pathlib.Path(project["output_root"]) / project["map_subdir"] / paste["run_name"]
    cat, nside, n = c["resolved_theory"]["catalog_key"], int(paste["nside"]), int(paste["num_splits"])
    ok = all((root / f"abacus_pasted_maps_{cat}_nside{nside}_split{s:03d}of{n:03d}.h5.timing.json").is_file()
             for s in range(n))
    if ok:
        ready.append(entry)
print(f"      {len(ready)} of {len(plan['entries'])} points are complete")
if len(ready) < int(sys.argv[3]):
    raise SystemExit(f"      only {len(ready)} ready; MIN_POINTS={sys.argv[3]}. "
                     f"Nothing written. Resubmit later.")
out = dict(plan); out["entries"] = ready; out["count"] = len(ready)
out["note"] = "partial round-1 design: the subset of points whose paste had finished"
pathlib.Path(sys.argv[2]).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
INNER

echo "### step 2: combine any ready point that is not combined yet"
$PY - "$WORK/ready_plan.json" <<'INNER' > "$WORK/to_combine.txt"
import json, pathlib, sys, yaml
REPO = pathlib.Path("/mnt/ceph/users/spandey/ltu-godmax/GODMAX")
plan = json.loads(pathlib.Path(sys.argv[1]).read_text())
for entry in plan["entries"]:
    cfg = pathlib.Path(entry["config_path"])
    if not cfg.is_absolute():
        cfg = REPO / cfg
    c = yaml.safe_load(cfg.read_text())
    paste, project = c["pasting"], c["project"]
    root = pathlib.Path(project["output_root"]) / project["map_subdir"] / paste["run_name"]
    cat, nside = c["resolved_theory"]["catalog_key"], int(paste["nside"])
    combined = root / f"abacus_pasted_maps_{cat}_nside{nside}.h5"
    if not combined.is_file():
        print(f"{cfg} {cat} {nside} {int(paste['num_splits'])}")
INNER
N_COMB=$(wc -l < "$WORK/to_combine.txt")
echo "      $N_COMB point(s) need combining"

# Combine in parallel.  One point takes ~60 s and is dominated by reading 32 x ~90 MB of
# split HDF5 and summing three float32 maps, so it is IO-bound and single-threaded: running
# COMB_JOBS at once turns a 70-minute serial pass into ~9 minutes.  --overwrite because
# combine_partial_maps writes the final HDF5 IN PLACE with no atomic rename, so an existing
# file is not proof of a complete one; redoing it is 60 s and removes the hazard.
COMB_JOBS=${COMB_JOBS:-16}
cat > "$WORK/combine_one.sh" <<COMBEOF
#!/bin/bash
set -euo pipefail
echo "      combining \$(basename "\$1")"
$PY $REPO/notebooks/xDESI/combine_abacus_map_splits.py \
    --config "\$1" --catalog "\$2" --num-splits "\$4" --nside "\$3" --overwrite
COMBEOF
chmod +x "$WORK/combine_one.sh"
if [ "$N_COMB" -gt 0 ]; then
    xargs -a "$WORK/to_combine.txt" -P "$COMB_JOBS" -n 4 "$WORK/combine_one.sh"
fi

echo "### step 3: measure the ready points through the frozen estimator"
cd "$REPO/notebooks/SBI_validate"
$PY measure_mock_sbi_pastes.py --plan "$WORK/ready_plan.json" \
    --output "$WORK/responses_$TAG.npz" --allow-missing

echo "### step 4: augment (no maps written -- this is a trial, the Cls are what matter)"
$PY build_mock_sbi_training_set.py --responses "$WORK/responses_$TAG.npz" \
    --output "$WORK/training_set_$TAG.npz" --replicas "$REPLICAS" --write-maps 0

echo "### step 5: NLE, pooled over $SEEDS network seeds"
$PY run_mock_sbi_snle.py --training-set "$WORK/training_set_$TAG.npz" \
    --output-dir "$WORK" --seeds "$SEEDS" --tag "$TAG"

echo "### done  $(date +%H:%M:%S)"
