#!/bin/bash
#SBATCH --account=bdne-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --mem=400G
#SBATCH --output=/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/run_scripts/logs/ref_%j.out

# --- Configuration ---
TOTAL_SPLITS=1
JDEVICE=0
NSIDE=512
WORK_DIR="/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new"

echo "Running Reference Job: Split ${JDEVICE} of ${TOTAL_SPLITS}"

# Load basic Cray modules for driver support
module purge
module load gcc-native/14
module load cudatoolkit/25.5_12.9

# Activate conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate godmax_env

# Priority 1: Conda Libraries (where you just installed libcudnn 9.8.0)
# Priority 2: Cray toolkit (for PTX/driver hooks)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CRAY_CUDATOOLKIT_DIR/lib64:$LD_LIBRARY_PATH
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CRAY_CUDATOOLKIT_DIR"

cd "${WORK_DIR}"

python run_bl.py \
    --nside ${NSIDE} \
    --jdevice ${JDEVICE} \
    --ndevices ${TOTAL_SPLITS} \
    --is_reference
