#!/bin/bash
#SBATCH --job-name=sc_oracle
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=data/SBI_validate/mock_sbi/logs/sc_oracle_%j.out
#SBATCH --error=data/SBI_validate/mock_sbi/logs/sc_oracle_%j.err
#
# Stage-0 oracle on a SELF-CONSISTENT pasted observation.  NO PASTES; this job is the
# go/no-go for the ~900 GPU-hour mock-SBI campaign.
#
# It answers: at 4 rounds of 128 pasted points with M free noise draws each, how close
# does the mock-SBI posterior get to the posterior that the same data and the same
# simulator actually imply?
#
# Why the completed bakeoff (oracle_bakeoff_*.json) does NOT answer this:
#   * it scored a MISSPECIFIED problem -- analytic stand-in against the pasted+noise
#     observation, chi2 floor 218.67 at the generating point;
#   * it scored against hmc_v2/run01, a GATE-REJECTED chain (r_hat 1.0127, 72.2%
#     tree-depth saturation) on a different observation;
#   * its noise was L @ epsilon rather than the measured field-level bank.
# This job fixes all three.  Its stand-in is mu_standin(u) = r_hat * mu_theory(u) against
# the noiseless pasted observation, so chi2 at the generating point is ~1e-26.
#
# Reference: self-normalised importance sampling on the exact stand-in likelihood.  One
# forward evaluation per draw instead of one per walker per step -- 60k instead of 3.8M
# for the equivalent emcee -- and it certifies itself through Pareto k.  A run whose
# reference reports k > 0.7 must be discarded, not interpreted.

set -euo pipefail
REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
OUT=$REPO/data/SBI_validate/mock_sbi
export JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=/tmp/matplotlib
cd "$REPO"
mkdir -p "$OUT/logs"

SCRIPT=notebooks/SBI_validate/07_mock_sbi_snle/oracle_mock_selfconsistent_test.py
LADDER="128 128 128 128"

echo "### step 1: exact reference (importance sampling, 60k draws)"
$PY -u $SCRIPT --arm exact --reference-method importance --reference-draws 60000 \
    --rebuild-reference

# Two arms at a time: each holds ~15 GiB of JAX allocation for its whole run, and the
# GPU is shared.  Four concurrent would exceed 80 GiB alongside another user.
run_pair () {
    $PY -u $SCRIPT --arm "$1" --compression "$2" --noise-per-theta "$3" \
        --theta-per-round $LADDER --max-wall-seconds 3000 &
    local a=$!
    $PY -u $SCRIPT --arm "$4" --compression "$5" --noise-per-theta "$6" \
        --theta-per-round $LADDER --max-wall-seconds 3000 &
    local b=$!
    wait $a $b
}

echo "### step 2: the ladder, four waves of two"
run_pair npe  score 64   snle score 64
run_pair npe  score 5    snle score 5
run_pair npe  raw   64   snre score 64
run_pair npe  raw   5    snre score 5

echo "### done; verdicts:"
$PY - <<'PYEOF'
import glob, json
rows = []
for path in sorted(glob.glob("data/SBI_validate/mock_sbi/oracle_sc_*.json")):
    d = json.load(open(path))
    if "rounds" not in d:
        continue
    for r in d["rounds"]:
        if r.get("status") != "trained":
            continue
        rows.append((d["arm"], d["compression"], d["noise_per_theta"],
                     r["cumulative_expensive_points"],
                     r["posterior"]["max_mean_drift_sigma"],
                     r["posterior"]["max_abs_width_relative_change"],
                     r["posterior_robust"]["max_median_drift_robust_sigma"],
                     r["posterior_robust"]["max_abs_robust_width_relative_change"]))
print(f"{'arm':6s} {'comp':6s} {'M':>4s} {'pts':>5s} {'drift':>7s} {'width':>8s} "
      f"{'r-drift':>8s} {'r-width':>8s}")
for row in sorted(rows, key=lambda r: (r[4],)):
    print(f"{row[0]:6s} {row[1]:6s} {row[2]:4d} {row[3]:5d} {row[4]:7.3f} {row[5]:+8.3f} "
          f"{row[6]:8.3f} {row[7]:+8.3f}")
print("\ngate: drift <= 0.10 sigma, width <= 0.10")
PYEOF
