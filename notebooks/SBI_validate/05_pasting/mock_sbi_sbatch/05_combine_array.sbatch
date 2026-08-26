#!/bin/bash
#SBATCH --job-name=msbi_combine
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/combine_%A_%a.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/SBI_validate/mock_sbi/logs/combine_%A_%a.err
#
# Combine the splits of one theta into one map product.  One array task per theta.
# CPU only.  require_strict_combine_contract is on, so this also verifies that every
# split shares one config sha, one catalog identity and one map schema -- which is the
# check that catches a half-repasted point.
#
#   PLAN=.../paste_plan.json sbatch --array=0-11 05_combine_array.sbatch

set -euo pipefail
REPO=/mnt/ceph/users/spandey/ltu-godmax/GODMAX
PY=/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python
: "${PLAN:?set PLAN to the paste_plan.json}"
export JAX_ENABLE_X64=True MPLCONFIGDIR=/tmp/matplotlib
cd "$REPO"

CFG=$($PY - "$PLAN" "$SLURM_ARRAY_TASK_ID" <<'INNER'
import json, pathlib, sys
plan = json.loads(pathlib.Path(sys.argv[1]).read_text())
path = pathlib.Path(plan["entries"][int(sys.argv[2])]["config_path"])
if not path.is_absolute():
    path = pathlib.Path("/mnt/ceph/users/spandey/ltu-godmax/GODMAX") / path
print(path)
INNER
)
CAT=$($PY -c "import yaml; print(yaml.safe_load(open('$CFG'))['resolved_theory']['catalog_key'])")
NSIDE=$($PY -c "import yaml; print(int(yaml.safe_load(open('$CFG'))['pasting']['nside']))")
NUM_SPLITS=$($PY -c "import yaml; print(int(yaml.safe_load(open('$CFG'))['pasting']['num_splits']))")

echo "combining theta $SLURM_ARRAY_TASK_ID  cfg $(basename "$CFG")  splits $NUM_SPLITS"
$PY notebooks/xDESI/combine_abacus_map_splits.py --config "$CFG" --catalog "$CAT" \
    --num-splits "$NUM_SPLITS" --nside "$NSIDE"
