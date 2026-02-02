#!/bin/bash
#SBATCH --account=bdne-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --time=00:30:00        # Each split gets its own 30 min window
#SBATCH --gpus=1
#SBATCH --mem=400G
#SBATCH --mail-type=FAIL       # Only email on failure to avoid 1200 emails!
#SBATCH --mail-user=anshumana@berkeley.edu
#SBATCH --array=0-3          # 1 sample * 4 splits = 4 tasks
#SBATCH --output=/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/run_scripts/logs/ref_%A_%a.out

# Configuration
NSIDE=512
TOTAL_SPLITS=4
WORK_DIR="/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin"

# --- 2D Array Logic ---
SAMPLE_ID=$((SLURM_ARRAY_TASK_ID / TOTAL_SPLITS))
JDEVICE=$((SLURM_ARRAY_TASK_ID % TOTAL_SPLITS))

echo "Task ${SLURM_ARRAY_TASK_ID}: Sample ${ID}, Split ${JDEVICE} of ${TOTAL_SPLITS}"

module purge
module load gcc-native/12.3 cuda/12.2.0 cudnn/9.3.0.75
cd "${WORK_DIR}"

conda run -n godmax_env python run_backlight_reference.py \
    --nside 512 --jdevice ${JDEVICE} --ndevices 4 --is_reference
