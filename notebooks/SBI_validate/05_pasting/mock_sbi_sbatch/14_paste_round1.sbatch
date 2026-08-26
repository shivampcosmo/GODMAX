#!/bin/bash
#SBATCH --job-name=msbi_paste_r1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-127%24
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/paste_r1_%A_%a.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/paste_r1_%A_%a.err
#
# Mock-SBI round 1: paste 128 parameter points drawn from the validated design mixture.
#
# CONCURRENCY IS CAPPED AT 24 GPUs by the '%24' on the array directive, together with
# '--gpus=1' per task: at most 24 tasks are ever RUNNING, each holding exactly one GPU.
#
# 24, not the 32 requested, because the cluster's 'gpu' QOS enforces MaxJobsPU=24 and
# MaxTRESPU gres/gpu=24 per user. Asking for %32 is not an error -- Slurm simply holds the
# excess with reason QOSMaxJobsPerUserLimit -- but it makes squeue look like something is
# wrong when it is not, so the throttle is set to the real ceiling. Do not raise it.
# Nothing downstream (combine, measure, augment) requests a GPU at all.
#
# ONE ARRAY TASK = ONE PARAMETER POINT = 32 SPLITS, run sequentially inside the task.
# The obvious alternative -- one task per (point, split) -- would need 128*32 = 4096
# array entries, and this cluster's MaxArraySize is 1001. Looping inside the task also
# keeps 04_paste_array.sbatch's per-split invocation of paste_abacus_maps.py byte-for-byte
# unchanged, which matters: that path is the one whose y/tau/kappa output was null-tested
# BITWISE against the archived reference split, both with the theta override set to the
# frozen values and with get_galmap disabled.
#
# Cost, from measurement rather than projection. The galaxy-skip null test measured 196.6 s
# per split (against 508.6 s with the galaxy map on), i.e. 1.75 GPU-hours per parameter
# point. So one task ~= 32 * 197 s ~= 1.75 h and the round is 224 GPU-hours; at the QOS
# ceiling of 24 concurrent GPUs that is ~9.3 h of wall clock.
#
# The task is RESUMABLE: a split whose output already exists is skipped. A wall-clock kill
# or a node failure therefore costs only the split in flight, not the point.

set -uo pipefail
REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
PLAN=${PLAN:-$REPO/data/SBI_validate/mock_sbi/paste_plan/round_1/paste_plan.json}
export JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=/tmp/matplotlib
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$REPO"

INDEX=${SLURM_ARRAY_TASK_ID}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Resolve this point from the plan, and refuse to run if the config has changed since the
# plan was written: the plan's log_q is only valid for the theta in that exact file.
read -r CFG RUN_NAME CAT NSIDE NSPLIT OUTDIR < <($PY - "$PLAN" "$INDEX" <<'INNER'
import json, pathlib, sys, yaml, hashlib
REPO = pathlib.Path("/mnt/ceph/users/spandey/ltu-godmax/GODMAX")
plan = json.loads(pathlib.Path(sys.argv[1]).read_text())
entry = plan["entries"][int(sys.argv[2])]
cfg = pathlib.Path(entry["config_path"])
if not cfg.is_absolute():
    cfg = REPO / cfg
digest = hashlib.sha256()
with cfg.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 24), b""):
        digest.update(chunk)
if digest.hexdigest() != entry["config_sha256"]:
    raise SystemExit(f"FATAL: {cfg} changed since the plan was written")
c = yaml.safe_load(cfg.read_text())
paste, project = c["pasting"], c["project"]
root = pathlib.Path(project["output_root"]) / project["map_subdir"] / paste["run_name"]
print(cfg, paste["run_name"], c["resolved_theory"]["catalog_key"],
      int(paste["nside"]), int(paste["num_splits"]), root)
INNER
) || { echo "FATAL: could not resolve plan entry $INDEX"; exit 1; }

echo "=== point $INDEX  run_name $RUN_NAME  nside $NSIDE  splits $NSPLIT  $(date +%H:%M:%S)"
STARTED=$(date +%s)
DONE=0; SKIPPED=0
for SPLIT in $(seq 0 $((NSPLIT - 1))); do
    OUT=$(printf "%s/abacus_pasted_maps_%s_nside%d_split%03dof%03d.h5" \
          "$OUTDIR" "$CAT" "$NSIDE" "$SPLIT" "$NSPLIT")
    # Readiness is the .timing.json sidecar, NOT the .h5 itself. The paste writes the HDF5
    # in place with no atomic rename, so a task killed mid-split leaves a TRUNCATED .h5
    # that a mere existence test would happily skip on resubmission -- and a truncated
    # split feeds straight into the combine, which is a wrong number with no error. The
    # sidecar is written with tmp+os.replace AFTER the HDF5 is closed
    # (abacus_pasting_helpers._write_json_atomic), so it exists only if the split finished.
    if [ -s "$OUT.timing.json" ]; then
        SKIPPED=$((SKIPPED + 1)); continue
    fi
    [ -e "$OUT" ] && echo "  split $SPLIT has a .h5 but no timing sidecar: truncated, repasting"
    $PY notebooks/xDESI/paste_abacus_maps.py --config "$CFG" --catalog "$CAT" \
        --split-index "$SPLIT" --num-splits "$NSPLIT" --nside "$NSIDE"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "FATAL: split $SPLIT of point $INDEX exited $rc"; exit $rc
    fi
    DONE=$((DONE + 1))
done
ELAPSED=$(( $(date +%s) - STARTED ))
echo "=== point $INDEX complete: $DONE pasted, $SKIPPED already present, ${ELAPSED}s  $(date +%H:%M:%S)"
