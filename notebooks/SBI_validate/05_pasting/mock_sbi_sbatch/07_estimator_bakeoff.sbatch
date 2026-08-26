#!/bin/bash
#SBATCH --job-name=msbi_bakeoff
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/bakeoff_%A_%a.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/bakeoff_%A_%a.err
#
# Decide the mock-SBI methodology and the paste budget WITHOUT pasting anything.
# No GPU.  Five arms, one per array task, so the wall clock is the slowest single arm
# rather than the sum.  Each arm additionally carries a hard internal budget, so 6 hours
# is a guarantee and not an estimate.
#
# ---------------------------------------------------------------------------------
# WHAT IS BEING TESTED
#
# The mock simulator is  x = mu_paste(theta) + nu(seed), and the noise adds exactly
# linearly (verified to 8e-15), so noise draws are free and unlimited.  The only
# expensive thing is mu_paste(theta).  The analytic model mu_theory(theta) is the same
# kind of object -- a smooth deterministic 5-parameter map to a 42-vector -- and costs
# 21 ms, so it stands in for the paste and the whole question becomes free to ask.
#
# Each arm: draw N distinct theta from the x2-inflated theory posterior (capped at the
# prior), evaluate the stand-in simulator, add free noise draws, train, sample, and score
# against the archived 10,000-sample NUTS chain of that same model on that same
# observation.  Simulator and reference are the same model, so any disagreement is the
# inference machinery, not physics.  Rounds draw from the previous posterior, so
# sequential sharpening is exercised too.
#
# Gates: mean drift <= 0.10 sigma, width <= 10%, correlation <= 0.10.  Both moment and
# tail-robust (normalized IQR) versions are reported, because a plain standard deviation
# was NOT reproducible on these posteriors: the emulator arm gave width +4.356 and +0.300
# at the same N on different sampler seeds.
#
# ---------------------------------------------------------------------------------
# WHAT WAS CUT, AND ON WHAT EVIDENCE
#
# * raw 42-vector arms: direct NPE on raw 42-D drifted 7.269 sigma at 96 points where the
#   exact 5-D score drifted 1.011, and the pooled variant stayed 49-64% too narrow at
#   every round.  Dropped on measurement.
# * SNPE-C arm: its atomic loss dies with NaN/Inf at round 2 in both compressions, the
#   same failure the theory campaign hit at 65,536 simulations per round.  Decided.
# * pooled p0/q reweighting NPE arm: the correction collapses exactly as the rounds
#   sharpen (ESS 70% -> 1.3%, Pareto k +0.51 -> +1.09).  Decided.
# * the GPU: nothing here needs one.
#
# ---------------------------------------------------------------------------------
# THE FIVE ARMS
#
#   0  SNRE  n=64  seed 0   ladder 48,48,96,96  -> rungs at 48 / 96 / 192 / 288
#   1  SNLE  n=64  seed 0   ladder 48,48,96,96     Arms 0 and 1 are the methodology
#                                                  choice.  SNLE has the best single
#                                                  result of any arm so far (0.484 sigma
#                                                  / +29% at 96 points); SNRE's MCMC is
#                                                  10x cheaper.
#   2  SNLE  n=256 seed 0   ladder 96,96,96     -> rows versus distinct theta.  Four
#                                                  times the training rows at the same
#                                                  theta count.  If that helps, the
#                                                  binding constraint is rows and the
#                                                  paste count goes further than
#                                                  expected; if it does not, distinct
#                                                  theta is binding and the paste budget
#                                                  is what must grow.  THIS IS THE ARM
#                                                  THAT ANSWERS "how many points".
#                                                  Compared against arms 0/1 at the
#                                                  shared rungs 96 / 192 / 288.
#   3  SNRE  n=64  seed 1   ladder 48,48,96,96  -> reproducibility.  Confirmed to be a
#   4  SNLE  n=64  seed 1   ladder 48,48,96,96     live risk: identical SNRE settings at
#                                                  24 points gave 4.572 sigma on one seed
#                                                  and 12.729 on another.
#
# The unequal 48,48,96,96 ladder is deliberate: it keeps resolution at the low end, where
# the budget decision actually lives, while costing four trainings instead of six.
#
# ---------------------------------------------------------------------------------
# AFTER IT FINISHES
#   bash 07_estimator_bakeoff.sbatch --summary      (prints the comparison table)

set -uo pipefail
REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
OUT=$REPO/data/SBI_validate/mock_sbi
cd "$REPO/notebooks/SBI_validate"

# --summary: collate whatever is on disk, runnable any time, including mid-flight.
if [ "${1:-}" = "--summary" ]; then
  $PY - <<'PYEOF'
import json, glob, pathlib
files = sorted(glob.glob("/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/oracle_bakeoff_*.json"))
if not files:
    raise SystemExit("no bakeoff results yet")
print(f"{'arm':24s} {'rnd':>3} {'pts':>4} {'rows':>6} {'drift':>7} {'width':>8} "
      f"{'r-drift':>8} {'r-width':>8} {'sec':>6}  gate")
for f in files:
    d = json.load(open(f))
    name = pathlib.Path(f).stem.replace("oracle_bakeoff_", "")
    for r in d["rounds"]:
        if r["status"] != "trained":
            print(f"{name:24s} {r['round']:>3}  ---- {r['status']}: {r.get('error','')[:60]}")
            continue
        m, rb = r["posterior"], r["posterior_robust"]
        print(f"{name:24s} {r['round']:>3} {r['cumulative_expensive_points']:>4} "
              f"{r['training_rows_this_round']:>6} {m['max_mean_drift_sigma']:>7.3f} "
              f"{m['max_abs_width_relative_change']:>+8.3f} "
              f"{rb['max_median_drift_robust_sigma']:>8.3f} "
              f"{rb['max_abs_robust_width_relative_change']:>+8.3f} "
              f"{r['round_seconds']:>6.0f}  {'PASS' if r['gate_passed'] else 'fail'}")
    if d.get("stopped_early"):
        s = d["stopped_early"]
        print(f"{name:24s}      truncated before round {s['before_round']} "
              f"({s['elapsed_seconds']:.0f}s elapsed, projected {s['projected_seconds']:.0f}s)")
print("\ngates: drift <= 0.10 sigma, width <= 10%")
print("arm 2 (n256) against arms 0/1 at rungs 96/192/288: if n256 is better at equal")
print("points, training rows bind rather than the number of pastes.")
PYEOF
  exit 0
fi

# 30 minutes of the 6 hour wall held back for startup, contention and finalisation.
BUDGET=19800

case "$SLURM_ARRAY_TASK_ID" in
  0) EST=snre; NOISE=64;  SEED=0; LADDER="48 48 96 96" ;;
  1) EST=snle; NOISE=64;  SEED=0; LADDER="48 48 96 96" ;;
  2) EST=snle; NOISE=256; SEED=0; LADDER="96 96 96"    ;;
  3) EST=snre; NOISE=64;  SEED=1; LADDER="48 48 96 96" ;;
  4) EST=snle; NOISE=64;  SEED=1; LADDER="48 48 96 96" ;;
  *) echo "unexpected array index $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac
TAG="bakeoff_${EST}_n${NOISE}_s${SEED}"

export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "arm $SLURM_ARRAY_TASK_ID: $TAG  ladder='$LADDER'  budget ${BUDGET}s  $(date -Is)"
$PY oracle_snle_snre_test.py \
    --estimator "$EST" --compression score \
    --theta-per-round $LADDER --noise-per-theta "$NOISE" --seed-offset "$SEED" \
    --posterior-samples 8000 --max-wall-seconds "$BUDGET" \
    --output-dir "$OUT" --tag "$TAG"
rc=$?
echo "arm $SLURM_ARRAY_TASK_ID: $TAG rc=$rc  $(date -Is)"
# rc=1 means this arm did not clear the gate, which is a result, not a job failure.
exit 0
