#!/bin/bash
#SBATCH --job-name=sc_snre
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/sc_snre_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/sc_snre_%j.err
#
# Stage-0 oracle, SNRE arm, on the SELF-CONSISTENT pasted observation.  NO PASTES.
# This is part of the go/no-go for the ~900 GPU-hour mock-SBI campaign.
#
# Why SNRE rather than NPE.  The NPE ladder was measured on exactly this problem and it
# is non-monotonic, because its proposal correction collapses.  Pareto k by round was
# +0.10 / +1.18 / +0.50 / +0.78 at 128 / 256 / 384 / 512 points; k > 0.7 means the
# importance weights have no finite variance, so rounds 2 and 4 are artifacts rather
# than posteriors.  The best VALID round was 384 points at 0.266 sigma / +23% -- and 512
# points was WORSE than 384.  SNRE takes no proposal argument at all: the prior is
# applied analytically when the posterior is sampled, so the design distribution can only
# change where the surrogate is accurate, never what is being estimated.  That failure
# mode cannot occur here.
#
# Resources.  The work splits across both devices and needs both:
#   * the JAX forward model runs on the GPU at 33.8 ms per evaluation (measured, while
#     sharing the card).  The same model on CPU could not finish 2 rounds of 16 points
#     in 2 minutes, so a CPU-only job is not viable.
#   * sbi trains on the CPU (device="cpu" in the arm constructors), which is what the 16
#     cores are for.  Moving training to the GPU would need --torch-device cuda plumbed
#     through and is NOT tested; do not assume it works.
# Measured end-to-end for reference: the full NPE ladder -- 4 rounds, 512 points, 32,768
# training rows -- took 19 minutes on one shared A100.  SNRE round 1 alone took 615 s.
#
# Both jobs score against the ONE cached reference (oracle_sc_exact_reference.npz,
# Pareto k +0.398, ESS 451/10000) and the ONE cached MAP/score operator
# (oracle_sc_reference_point.npz).  Do NOT pass --rebuild-reference or
# --rebuild-reference-point: arms scored against different references are not comparable
# to each other, nor to the NPE ladder already measured.
#
# Two seed offsets on the headline arm, because a single run is not reproducible: two
# identical-design SNLE runs gave 0.194 and 0.549 sigma drift at 128 points, differing
# only in the network and MCMC seed.

set -uo pipefail
REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
S=notebooks/SBI_validate/07_mock_sbi_snle/oracle_mock_selfconsistent_test.py
L=$REPO/data/SBI_validate/mock_sbi/logs
export JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=/tmp/matplotlib
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd "$REPO"
mkdir -p "$L"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for required in data/SBI_validate/mock_sbi/oracle_sc_exact_reference.npz \
                data/SBI_validate/mock_sbi/oracle_sc_reference_point.npz; do
    if [ ! -f "$required" ]; then
        echo "FATAL: $required is missing.  Build both with:"
        echo "  $PY $S --arm exact"
        exit 1
    fi
done

START=$(date +%s)
JOB_SECONDS=14400
RESERVE=600

# SLURM killing the process at the wall limit loses the final artifact and the verdict,
# leaving only a log.  So each arm is told how much time is actually left; a round cannot
# be stopped part way, so the runner declines to START one it projects cannot finish and
# exits cleanly with the ladder it did complete, recording stopped_early.  That makes the
# wall limit a guarantee rather than a rate extrapolation.
run_arm () {   # compression  noise_per_theta  seed_offset
    local now left tag
    now=$(date +%s)
    left=$(( JOB_SECONDS - (now - START) - RESERVE ))
    tag="snre_$1_m$2_s$3"
    if [ "$left" -lt 900 ]; then
        echo "### skipping $tag -- only ${left}s left"
        return 0
    fi
    echo "### snre $1 M=$2 seed=$3   budget ${left}s   $(date +%H:%M:%S)"
    $PY -u $S --arm snre --compression "$1" --noise-per-theta "$2" \
        --seed-offset "$3" --theta-per-round 128 128 128 128 \
        --forward-chunk 16 --max-wall-seconds "$left" \
        > "$L/sc_$tag.log" 2>&1
    echo "    exit $?   $(date +%H:%M:%S)"
    grep -E "cumulative|stopping|\[5/5\]" "$L/sc_$tag.log" || true
}

# Highest-value first, so even a truncated job answers the main question.
run_arm score 64 0
run_arm score 64 1
run_arm score 5  0
run_arm raw   64 0

echo
echo "### summary of every self-consistent oracle arm on disk"
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
        rows.append((d["arm"], d["compression"], d["noise_per_theta"], d["seed_offset"],
                     r["cumulative_expensive_points"],
                     r["posterior"]["max_mean_drift_sigma"],
                     r["posterior"]["max_abs_width_relative_change"],
                     r.get("reweighting_pareto_k")))
print(f"{'arm':5s} {'comp':6s} {'M':>3s} {'sd':>2s} {'pts':>5s} {'drift':>7s} {'width':>8s} {'k':>8s}")
for row in sorted(rows):
    k = "     n/a" if row[7] is None else f"{row[7]:+8.2f}"
    flag = "" if (row[7] is None or row[7] <= 0.7) else "  <- k>0.7, NOT a posterior"
    print(f"{row[0]:5s} {row[1]:6s} {row[2]:3d} {row[3]:2d} {row[4]:5d} "
          f"{row[5]:7.3f} {row[6]:+8.3f} {k}{flag}")
print()
print("gate: drift <= 0.10 sigma, width <= 0.10.")
print("k is n/a for SNLE/SNRE by design -- they need no proposal correction, so there is")
print("no importance weight to diagnose, which is the whole reason they are being tried.")
PYEOF
