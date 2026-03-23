#!/bin/bash
#SBATCH --account=bdne-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --gpus=1
#SBATCH --mem=400G
#SBATCH --mail-type=FAIL       # Only email on failure to avoid lots of emails!
#SBATCH --mail-user=anshumana@berkeley.edu
#SBATCH --array=0-499          # 500 samples
#SBATCH --output=/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/run_scripts/logs/LHS_%A_%a.out

# Configuration
NSIDE=512
TOTAL_SPLITS=1
WORK_DIR="/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new"
CSV_FILE="${WORK_DIR}/lhs_samples.csv"

# --- 2D Array Logic ---
SAMPLE_ID=$((SLURM_ARRAY_TASK_ID / TOTAL_SPLITS))
JDEVICE=$((SLURM_ARRAY_TASK_ID % TOTAL_SPLITS))

# Extract parameters for the Sample_ID (Row = ID + 2)
LINE_NUM=$((SAMPLE_ID + 2))
PARAMS=$(sed -n "${LINE_NUM}p" "$CSV_FILE")

# Parse CSV
ID=$(echo $PARAMS | cut -d',' -f1)
THETA_0=$(echo $PARAMS | cut -d',' -f2)
NU_THETA_EJ_M=$(echo $PARAMS | cut -d',' -f3)

echo "Task ${SLURM_ARRAY_TASK_ID}: Sample ${ID}, Split ${JDEVICE} of ${TOTAL_SPLITS}"

module purge
module load gcc-native/12.3 cuda/12.2.0 cudnn/9.3.0.75
cd "${WORK_DIR}"

conda run -n godmax_env bash -c "export PYTHONNOUSERSITE=1; LD_LIBRARY_PATH= python run_bl.py \
    --nside ${NSIDE} \
    --jdevice ${JDEVICE} \
    --ndevices ${TOTAL_SPLITS} \
    --theta_ej_0 ${THETA_0} \
    --nu_theta_ej_M ${NU_THETA_EJ_M} \
    --sample_id ${ID}"
