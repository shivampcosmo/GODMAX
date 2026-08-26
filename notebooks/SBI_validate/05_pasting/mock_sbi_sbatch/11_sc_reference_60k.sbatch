#!/bin/bash
#SBATCH --job-name=sc_ref60k
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/sc_ref60k_%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/sc_ref60k_%j.err
#
# A better exact reference for the self-consistent oracle.  No pastes.
#
# The 10k reference has ESS 451, which is 0.047 sigma of noise on means and 3.3% on
# widths -- about half the strict gate and a quarter of the relaxed one.  Drift values
# of 0.08-0.20 sigma cannot be resolved against that.  60k draws at the same Pareto k
# gives ESS ~2700: 0.019 sigma and 1.4%, which is a tenth of the relaxed gate.
#
# Warm-started from the existing 10k reference's own moments.  Starting from the Laplace
# instead is known to waste a pass: it measured Pareto k 0.887 / ESS 19 and only the
# adaptive refit rescued it, so a cold 60k pass would burn ~34 minutes reproducing a
# failure.  The proposal enters only through log q and divides out of the weights, so
# warm-starting changes efficiency and never the target.
#
# Written to a NEW path.  The 10k reference stays on disk because every arm measured so
# far was scored against it, and silently replacing it would make old and new numbers
# look comparable when they are not.

set -euo pipefail
REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
D=$REPO/data/SBI_validate/mock_sbi
export JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false MPLCONFIGDIR=/tmp/matplotlib
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

$PY -u notebooks/SBI_validate/07_mock_sbi_snle/oracle_mock_selfconsistent_test.py \
    --arm exact --reference-method importance \
    --reference-draws 60000 --reference-adapt-passes 2 \
    --reference-pareto-k-target 0.5 \
    --reference-warm-start "$D/oracle_sc_exact_reference.npz" \
    --reference-path "$D/oracle_sc_exact_reference_60k.npz" \
    --rebuild-reference --forward-chunk 16

echo "### the new reference must clear Pareto k 0.7 to be usable at all:"
$PY - <<'PYEOF'
import json, numpy as np
d = np.load("data/SBI_validate/mock_sbi/oracle_sc_exact_reference_60k.npz", allow_pickle=True)
m = json.loads(str(d["manifest_json"]))
ess = m["effective_sample_size"]
print(json.dumps({k: m[k] for k in
                  ("pareto_k", "effective_sample_size", "effective_sample_fraction",
                   "max_normalised_weight", "n_draws", "seconds")}, indent=2))
print(f"noise on means {1/np.sqrt(ess):.4f} sigma;  on widths {100/np.sqrt(2*ess):.2f}%")
print("VERDICT:", "USABLE" if m["pareto_k"] <= 0.7 else "NOT USABLE -- discard, do not interpret")
PYEOF
