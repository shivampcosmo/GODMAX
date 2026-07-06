#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --job-name=run_3_4_1024
#SBATCH -p gpu
#SBATCH -C a100-80gb
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --output=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/pasting//run_scripts/websky/logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/pasting//run_scripts/websky/logs/%x.%j.err

echo "--- JOB DETAILS ---"
echo "Job Name: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Running on: $(hostname)"
echo "NSIDE: 1024"
echo "JDEVICE: 3"
echo "TOTAL_DEVICES: 4"
echo "-------------------"

# Load necessary modules
module purge
module load python
source ~/miniconda3/bin/activate ili-sbi

# Change to the working directory
cd "/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/pasting/"

# Run the python script with the correct parameters
# The 'time' command will measure the execution duration
# time srun python run_nside_map_halfdome.py 1024 3 4
time srun python run_nside_tau_split_websky.py 1024 3 4

echo "done"
