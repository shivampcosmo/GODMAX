#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --job-name=run_4_8_2048
#SBATCH -p gpu
#SBATCH -C a100-80gb
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --output=/mnt/home/spandey/ceph/paste_godmax/GODMAX/notebooks/pasting//run_scripts/logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/paste_godmax/GODMAX/notebooks/pasting//run_scripts/logs/%x.%j.err

echo "--- JOB DETAILS ---"
echo "Job Name: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Running on: $(hostname)"
echo "NSIDE: 2048"
echo "JDEVICE: 4"
echo "TOTAL_DEVICES: 8"
echo "-------------------"

# Load necessary modules
module purge
module load python
source ~/miniconda3/bin/activate ili-sbi

# Change to the working directory
cd "/mnt/home/spandey/ceph/paste_godmax/GODMAX/notebooks/pasting/"

# Run the python script with the correct parameters
# The 'time' command will measure the execution duration
time srun python run_nside_map_halfdome.py 2048 4 8

echo "done"
