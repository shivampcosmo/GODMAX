#!/usr/bin/env bash
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=sbi_3p_catalog
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/SBI_validate/logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/SBI_validate/logs/%x.%j.err

set -euo pipefail
umask 027

readonly REPO_ROOT="/mnt/ceph/users/spandey/ltu-godmax/GODMAX"
readonly PYTHON="/mnt/home/spandey/miniconda3/envs/ili-sbi/bin/python"
readonly CONFIG="${REPO_ROOT}/notebooks/SBI_validate/three_probe_mock_experiment.yaml"
readonly BUILDER="${REPO_ROOT}/notebooks/xDESI/abacus_lightcone_catalog.py"
readonly VALIDATOR="${REPO_ROOT}/notebooks/SBI_validate/01_catalog_and_theory_inputs/validate_three_probe_catalog_preflight.py"
readonly CATALOG_KEY="c0000_z0p3_0p5_mmin5e11"
readonly OUTPUT="${REPO_ROOT}/data/SBI_validate/three_probe_mock/catalogs/abacus_c0000_ph000_zgt0p3_zlt0p5_Mproxyge5e11_halos.h5"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${REPO_ROOT}"

/usr/bin/time -v "${PYTHON}" -u "${BUILDER}" \
  --config "${CONFIG}" \
  --catalog "${CATALOG_KEY}"

"${PYTHON}" "${VALIDATOR}" --config "${CONFIG}" --catalog-path "${OUTPUT}"
sha256sum "${OUTPUT}"
