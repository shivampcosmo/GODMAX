#!/bin/bash
#SBATCH --account=def-lplevass
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2                  # total number of GPUs

#SBATCH --cpus-per-task=1         # CPU cores per MPI process
#SBATCH --mem=80G          # host memory per CPU core
#SBATCH --time=0-02:59            # time (DD-HH:MM)
#SBATCH --job-name=NUTS_UP_run_deproj_cib_1p7_dBeta_probe_xipxim
#SBATCH --output=/project/def-lplevass/shivamp/GODMAX/run_scripts/narval/logs/%x.%j.out
#SBATCH --error=/project/def-lplevass/shivamp/GODMAX/run_scripts/narval/logs/%x.%j.err
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# module purge
module load python
module load scipy-stack
module load cuda
source /home/shivamp/godmax/bin/activate

cd /project/def-lplevass/shivamp/GODMAX/run_scripts/narval/
time srun python sample_params.py cib_1p7_dBeta xip_xim
echo "done"
