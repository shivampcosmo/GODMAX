#!/bin/bash
#SBATCH --job-name=tsz_map_jax_dist
#SBATCH --output=/mnt/home/spandey/ceph/paste_godmax/GODMAX/notebooks/pasting/run_scripts/logs/tsz_map_%j.out
#SBATCH --error=/mnt/home/spandey/ceph/paste_godmax/GODMAX/notebooks/pasting/run_scripts/logs/tsz_map_%j.err
#SBATCH --nodes=3                    # Number of nodes
#SBATCH --ntasks-per-node=1          # VERY IMPORTANT: one JAX process per node
#SBATCH --gpus-per-node=4            # GPUs per node (your python script will manage these with pmap)
#SBATCH --cpus-per-gpu=8             # Allocate enough CPUs for data pre-processing (multiprocessing.Pool)
#SBATCH --time=01:30:00              # Wall time limit
#SBATCH -p gpu
#SBATCH -C a100-80gb
#SBATCH --mem=0                      # Let SLURM allocate memory based on CPUs/GPUs for the whole node

# 1. SETUP ENVIRONMENT
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Number of nodes: $SLURM_NNODES"
echo "Node list: $SLURM_NODELIST"
echo "========================================================"

# Load necessary modules and activate conda environment
module purge
module load python
# module load cuda
# module load cudnn
source ~/miniconda3/bin/activate ili-sbi


# Set JAX/XLA environment variables for performance
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# Optional: performance flags for NVIDIA GPUs
# export XLA_FLAGS="--xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_collectives=true"


# 2. DEFINE PARAMETERS
NSIDE=2048  # Set your desired nside here
PYTHON_SCRIPT="tsz_map_distributed3.py" # Your python script from the previous step

# Navigate to the script directory
cd /mnt/home/spandey/ceph/paste_godmax/GODMAX/notebooks/pasting/
echo "Current directory: $(pwd)"
echo "Running script: ${PYTHON_SCRIPT} with NSIDE=${NSIDE}"
echo "========================================================"


# 3. LAUNCH THE DISTRIBUTED JOB WITH SRUN
# srun will launch one instance of this command on each of the --nodes you requested.
# JAX's `jax.distributed.initialize()` will automatically detect the SLURM environment
# and configure the communication between the nodes. No manual setup is needed.
# The "-u" flag for python is for unbuffered output, which is useful for seeing logs in real-time.

srun python -u $PYTHON_SCRIPT $NSIDE

echo "========================================================"
echo "Job finished with exit code $? at $(date)."
echo "========================================================"