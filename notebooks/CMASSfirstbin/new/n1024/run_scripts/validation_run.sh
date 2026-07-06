#!/bin/bash
#SBATCH --account=bdne-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --mem=400G
#SBATCH --mail-type=FAIL       # Only email on failure to avoid lots of emails!
#SBATCH --mail-user=anshumana@berkeley.edu
#SBATCH --array=0-49          # 50 samples
#SBATCH --output=/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/run_scripts/logs/val_%A_%a.out

# Configuration
NSIDE=1024
TOTAL_SPLITS=1
WORK_DIR="/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024"
CSV_FILE="${WORK_DIR}/validation_samples.csv"

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


echo "Task ${SLURM_ARRAY_TASK_ID}: validation sample Index ${SAMPLE_ID}, Split ${JDEVICE} of ${TOTAL_SPLITS}"

module purge
module load gcc-native/14 cuda/12.2.0 cudnn/9.3.0.75
cd "${WORK_DIR}"

conda run -n godmax_env bash -c "export PYTHONNOUSERSITE=1; LD_LIBRARY_PATH= python run_bl.py \
    --nside ${NSIDE} \
    --jdevice ${JDEVICE} \
    --ndevices ${TOTAL_SPLITS} \
    --is_validation \
    --theta_ej_0 ${THETA_0} \
    --nu_theta_ej_M ${NU_THETA_EJ_M} \
    --sample_id ${SAMPLE_ID}"
