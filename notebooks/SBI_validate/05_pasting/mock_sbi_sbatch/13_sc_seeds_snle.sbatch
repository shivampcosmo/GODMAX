#!/bin/bash
#SBATCH --job-name=sc_seed_snle
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-3%4
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/sc_seed_snle_%A_%a.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/sc_seed_snle_%A_%a.err
#
# SNLE seed replicates for the self-consistent oracle.  NO PASTES.
#
# Why replicates rather than one more configuration.  Two SNLE-family runs with
# IDENTICAL designs, differing only in the network and MCMC seed, gave 0.078 and 0.522
# sigma mean drift at 512 pasted points.  The seed scatter is therefore larger than the
# gate itself, and a single-seed number is seed luck reported as a result.  Four seeds
# put a standard error of roughly scatter/2 on each configuration's drift, which is what
# makes a 0.20 sigma verdict decidable.
#
# One array task per (compression, M, seed), one GPU each, so four seeds run
# simultaneously instead of in sequence: ~4 tasks in the wall time of one.
#
# Two deliberate choices:
#  * --save-samples, so the seeds can be POOLED into a deep-ensemble posterior and so any
#    arm can be rescored against a better reference without re-running it.
#  * runs score against the EXISTING 10k reference at run time, and are rescored against
#    the 60k reference afterwards by analyze_mock_sbi_seed_ensemble.py.  Drift, width and
#    correlation change are pure functions of the stored first and second moments, so the
#    rescoring is exact -- which is why this job does NOT wait on job 11 and the whole
#    set finishes in one wall time rather than two.
#
# Output goes to seed_ensemble/ so the earlier single-seed records are not overwritten;
# they are the evidence that the scatter exists.

set -uo pipefail
REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
S=notebooks/SBI_validate/07_mock_sbi_snle/oracle_mock_selfconsistent_test.py
OUT=$REPO/data/SBI_validate/mock_sbi/seed_ensemble
export JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=/tmp/matplotlib
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$REPO"
mkdir -p "$OUT"

for required in data/SBI_validate/mock_sbi/oracle_sc_exact_reference.npz \
                data/SBI_validate/mock_sbi/oracle_sc_reference_point.npz; do
    [ -f "$required" ] || { echo "FATAL: $required missing; run $PY $S --arm exact"; exit 1; }
done

# task index -> (compression, noise_per_theta, seed)
CONFIGS=()
case snle in
  snre) SPECS=("score 64" "raw 64" "score 5") ;;
  snle) SPECS=("score 64") ;;
esac
N_SEEDS=4
IDX=${SLURM_ARRAY_TASK_ID}
SPEC_INDEX=$(( IDX / N_SEEDS ))
SEED=$(( IDX % N_SEEDS ))
read -r COMP M <<< "${SPECS[$SPEC_INDEX]}"

echo "task $IDX -> snle compression=$COMP M=$M seed=$SEED   $(date +%H:%M:%S)"
nvidia-smi --query-gpu=name --format=csv,noheader || true

# The wall limit must never be what stops a round: SLURM killing the process loses the
# artifact and the verdict.  The runner declines to START a round it projects cannot
# finish and exits cleanly recording stopped_early.
BUDGET=$(( $(echo 01:30:00 | awk -F: '{print $1*3600+$2*60+$3}') - 600 ))

$PY -u $S --arm snle --compression "$COMP" --noise-per-theta "$M" \
    --seed-offset "$SEED" --theta-per-round 128 128 128 128 \
    --forward-chunk 16 --save-samples --gate-profile relaxed_20260825 \
    --max-wall-seconds "$BUDGET" --output-dir "$OUT"
echo "exit $?   $(date +%H:%M:%S)"
