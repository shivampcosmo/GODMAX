#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH -C rome
#SBATCH -p cmbas
#SBATCH --time=00:30:00
#SBATCH --job-name=measure_3_6_8
#SBATCH --output=/mnt/ceph/users/spandey/paste_godmax/GODMAX/notebooks/xDESI/run_scripts/slurm_scripts/slurm_scripts/logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/paste_godmax/GODMAX/notebooks/xDESI/run_scripts/slurm_scripts/slurm_scripts/logs/%x.%j.err

echo "--- JOB DETAILS ---"
echo "Job Name: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Running on: $(hostname)"
echo "JDEVICE: 3"
echo "I_START: 6"
echo "I_END: 8"
echo "-------------------"

module purge
module load openmpi/4.1.8
module load python
source ~/miniconda3/bin/activate ili-sbi

# Loop through i values for this device
for i in {6..7};
do
    echo $i;
    cd "/mnt/ceph/users/spandey/paste_godmax/GODMAX/notebooks/xDESI";
    echo "$PWD";
    time srun python measure_abacus_Clgg.py $i;
    echo "done";
done

echo "All runs complete for device 3"
