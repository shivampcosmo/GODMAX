#!/bin/bash
#SBATCH --account=bdne-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --time=01:30:00        # Each split gets its own 1.5h window
#SBATCH --gpus=1
#SBATCH --mem=400G
#SBATCH --mail-type=FAIL       # Only email on failure to avoid lots of emails!
#SBATCH --mail-user=anshumana@berkeley.edu
#SBATCH --array=0-119          # 30 samples * 4 splits = 120 tasks
#SBATCH --output=/work/hdd/bdne/aacharya2/GODMAX/notebooks/anshuman/run_scripts/logs/LHS_%A_%a.out

# Configuration
NSIDE=512
TOTAL_SPLITS=4
WORK_DIR="/work/hdd/bdne/aacharya2/GODMAX/notebooks/anshuman"
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
NU_M=$(echo $PARAMS | cut -d',' -f3)
NU_Z=$(echo $PARAMS | cut -d',' -f4)
MU_B=$(echo $PARAMS | cut -d',' -f5)

echo "Task ${SLURM_ARRAY_TASK_ID}: Sample ${ID}, Split ${JDEVICE} of ${TOTAL_SPLITS}"

module purge
module load gcc-native/12.3 cuda/12.2.0 cudnn/9.3.0.75
cd "${WORK_DIR}"

conda run -n godmax_env bash -c "export PYTHONNOUSERSITE=1; LD_LIBRARY_PATH= python run_backlight_pkdgrav_all_maps.py \
    --nside ${NSIDE} \
    --jdevice ${JDEVICE} \
    --ndevices ${TOTAL_SPLITS} \
    --theta_ej_0 ${THETA_0} \
    --nu_theta_ej_M ${NU_M} \
    --nu_theta_ej_z ${NU_Z} \
    --mu_beta ${MU_B} \
    --sample_id ${ID}"
