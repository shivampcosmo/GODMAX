#!/bin/bash
#SBATCH --account=def-lplevass
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4                  # total number of GPUs

#SBATCH --cpus-per-task=10         # CPU cores per MPI process
#SBATCH --mem=80G          # host memory per CPU core
#SBATCH --time=0-23:59            # time (DD-HH:MM)
#SBATCH --job-name=NUTS_UP_run_deproj_None_probe_all
#SBATCH --output=/project/def-lplevass/shivamp/GODMAX/run_scripts/narval/logs/%x.%j.out
#SBATCH --error=/project/def-lplevass/shivamp/GODMAX/run_scripts/narval/logs/%x.%j.err
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JAX_TRACEBACK_FILTERING=off

module purge
module load python
module load cuda
module load scipy-stack
source /home/shivamp/godmax/bin/activate

cd /project/def-lplevass/shivamp/GODMAX/run_scripts/narval/
time srun python sample_params_simple.py None all
echo "done"
