#!/bin/bash
#SBATCH --account=bdne-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --mem=400G
#SBATCH --output=/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/run_scripts/logs/ref_%j.out

# --- Configuration ---
TOTAL_SPLITS=1
JDEVICE=0
NSIDE=1024
WORK_DIR="/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024"

echo "Running Reference Job: Split ${JDEVICE} of ${TOTAL_SPLITS}"

# Load basic Cray modules for driver support
module purge
module load gcc-native/14 cuda/12.2.0 cudnn/9.3.0.75

cd "${WORK_DIR}"

conda run -n godmax_env bash -c "export PYTHONNOUSERSITE=1; LD_LIBRARY_PATH= python run_bl.py \
    --nside ${NSIDE} \
    --jdevice ${JDEVICE} \
    --ndevices ${TOTAL_SPLITS} \
    --is_reference"
