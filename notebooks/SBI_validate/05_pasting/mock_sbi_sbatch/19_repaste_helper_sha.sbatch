#!/bin/bash
#SBATCH --job-name=msbi_repaste
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-23%24
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/repaste_%A_%a.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/repaste_%A_%a.err
#
# Repair the 24 round-1 points whose 32 splits carry TWO different pasting_helper_sha256
# stamps, which combine_partial_maps refuses (cross-split contract attrs must agree).
#
# WHY THEY ARE MIXED: abacus_pasting_helpers.py was edited at 22:11 on 2026-08-25 to fix
# the galaxy-n(z) guard WHILE the paste array was still running.  The paste stamps
# _sha256_file(Path(__file__)) into every split, so points that were mid-flight at that
# moment carry the pre-edit sha on their earlier splits and the post-edit sha on their
# later ones.  The map CONTENT is unaffected -- the edit is a single hunk inside
# combine_partial_maps (helpers 2922-3114), and paste_abacus_maps.py imports only
# run_paste_split and xdesi_dir -- but the contract cannot know that, and weakening it to
# accept two shas would blind it to a real version skew later.  So: repaste.
#
# PRECONDITION, CHECKED BELOW AND FATAL IF VIOLATED:
#   1. no msbi_paste_r1 array task is still running (repasting under a live producer is
#      what created this problem in the first place)
#   2. abacus_pasting_helpers.py currently hashes to TARGET_SHA, i.e. it has been restored
#      to its pre-edit bytes.  Restore with:
#          cp <scratch>/helper_preedit/abacus_pasting_helpers.py notebooks/xDESI/
#      and put the fix back AFTER this job finishes -- combining stamps nothing, so the
#      fixed helper can be restored immediately and every combine still passes.
#
# One array task = one point.  Each repastes only its minority splits (6-23 of 32, 238
# total), after DELETING the existing .h5 and its .timing.json so the resume check in
# 14_paste_round1.sbatch cannot skip them.  Repasting is bitwise-reproducible here: the
# chunk seed is f(split_index, chunk_id), num_splits and the catalog row order are
# unchanged, and get_galmap is false so no HOD sampling happens at all.

set -uo pipefail
REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
MANIFEST=${MANIFEST:-$REPO/data/SBI_validate/mock_sbi/paste_plan/round_1/helper_sha_repaste.json}
TARGET_SHA=2044f85df2206c9b53bb9fa2ab5916dbdc0d8297d169ea27c073196640ea773e
export JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=/tmp/matplotlib
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$REPO"

# --- precondition 1: nothing else is pasting -------------------------------------------
LIVE=$(squeue -u "$USER" -h -n msbi_paste_r1 -t RUNNING,PENDING -o "%i" 2>/dev/null | wc -l)
if [ "$LIVE" -ne 0 ]; then
    echo "FATAL: $LIVE msbi_paste_r1 task(s) still queued or running."
    echo "       Repasting now would stamp splits while another producer is writing with a"
    echo "       different helper, which is exactly how the mixed stamps arose. Wait."
    exit 1
fi

# --- precondition 2: the helper is at the pre-edit bytes --------------------------------
HAVE=$($PY -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('notebooks/xDESI/abacus_pasting_helpers.py').read_bytes()).hexdigest())")
if [ "$HAVE" != "$TARGET_SHA" ]; then
    echo "FATAL: abacus_pasting_helpers.py hashes to $HAVE"
    echo "       expected $TARGET_SHA (the pre-edit bytes the surviving splits carry)."
    echo "       Restore it before running this job, or the repaste stamps the WRONG sha"
    echo "       and the point stays mixed."
    exit 1
fi
echo "=== helper sha verified $HAVE"

read -r CFG CAT NSIDE NSPLIT SPLITS < <($PY - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'INNER'
import json, pathlib, sys
m = json.loads(pathlib.Path(sys.argv[1]).read_text())
e = m["entries"][int(sys.argv[2])]
print(e["config_path"], e["catalog"], e["nside"], e["num_splits"],
      ",".join(str(s) for s in e["repaste_splits"]))
INNER
) || { echo "FATAL: could not resolve manifest entry $SLURM_ARRAY_TASK_ID"; exit 1; }

OUTDIR=$($PY -c "
import yaml,pathlib,sys
c=yaml.safe_load(pathlib.Path('$CFG').read_text())
print(pathlib.Path(c['project']['output_root'])/c['project']['map_subdir']/c['pasting']['run_name'])")

echo "=== repasting $(basename "$CFG")  splits $SPLITS  $(date +%H:%M:%S)"
FAILED=0
for SPLIT in ${SPLITS//,/ }; do
    OUT=$(printf "%s/abacus_pasted_maps_%s_nside%d_split%03dof%03d.h5" \
        "$OUTDIR" "$CAT" "$NSIDE" "$SPLIT" "$NSPLIT")
    rm -f "$OUT" "$OUT.timing.json"
    echo "  split $SPLIT  $(date +%H:%M:%S)"
    $PY notebooks/xDESI/paste_abacus_maps.py --config "$CFG" --catalog "$CAT" \
        --num-splits "$NSPLIT" --split-index "$SPLIT" --nside "$NSIDE" || FAILED=$((FAILED+1))
    [ -s "$OUT.timing.json" ] || { echo "  FATAL: split $SPLIT produced no sidecar"; FAILED=$((FAILED+1)); }
done

# --- verify the point is now internally uniform -----------------------------------------
$PY - "$OUTDIR" "$CAT" "$NSIDE" "$NSPLIT" "$TARGET_SHA" <<'INNER'
import h5py, pathlib, sys, collections
outdir, cat, nside, nsp, target = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
shas = collections.Counter()
for s in range(nsp):
    f = pathlib.Path(outdir) / f"abacus_pasted_maps_{cat}_nside{nside}_split{s:03d}of{nsp:03d}.h5"
    with h5py.File(f, "r") as h:
        shas[str(h.attrs["pasting_helper_sha256"])] += 1
if len(shas) != 1 or target not in shas:
    raise SystemExit(f"FATAL: point is still mixed: { {k[:8]: v for k, v in shas.items()} }")
print(f"      point is uniform at {target[:8]}... across {nsp} splits")
INNER
RC=$?
echo "=== done  failed_splits=$FAILED  verify_rc=$RC  $(date +%H:%M:%S)"
exit $(( FAILED + RC ))
