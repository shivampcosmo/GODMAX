#!/bin/bash
#SBATCH --account=bdne-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --gpus=1
#SBATCH --mem=400G
#SBATCH --mail-type=FAIL       # Only email on failure to avoid lots of emails!
#SBATCH --mail-user=anshumana@berkeley.edu
#SBATCH --array=0-199          # 200 samples
#SBATCH --output=/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/run_scripts/logs/R2_%A_%a.out

# Configuration
NSIDE=512
TOTAL_SPLITS=1
WORK_DIR="/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new"
CSV_FILE="${WORK_DIR}/round2_samples.csv"

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

# Offset ID: Round 1 was 0-499. Round 2 should be 500-699.
# We add 30 to the index to create unique folder names
FINAL_ID=$((SAMPLE_ID + 500))

echo "Task ${SLURM_ARRAY_TASK_ID}: R2 Sample Index ${SAMPLE_ID} (ID ${FINAL_ID}), Split ${JDEVICE} of ${TOTAL_SPLITS}"

module purge
module load gcc-native/14 cuda/12.2.0 cudnn/9.3.0.75
cd "${WORK_DIR}"

# Note: We pass --sample_id ${FINAL_ID} so it creates /sample_30, /sample_31, etc.
conda run -n godmax_env bash -c "export PYTHONNOUSERSITE=1; LD_LIBRARY_PATH= python run_bl.py \
    --nside ${NSIDE} \
    --jdevice ${JDEVICE} \
    --ndevices ${TOTAL_SPLITS} \
    --theta_ej_0 ${THETA_0} \
    --nu_theta_ej_M ${NU_THETA_EJ_M} \
    --sample_id ${FINAL_ID}"
