#!/bin/bash
#SBATCH --job-name=GODMAX_Bridge
#SBATCH --output=bridge_%j.log
#SBATCH --error=bridge_%j.err
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --time=04:00:00

# Activate your environment
conda activate godmax_env

source /u/aacharya2/.bashrc  # Load your profile
conda activate godmax_env

# Hard-set the paths that made it work previously
export CONDA_LIB=$CONDA_PREFIX/lib
export LD_PRELOAD=$CONDA_LIB/libstdc++.so.6
export LD_LIBRARY_PATH=$CONDA_LIB:$LD_LIBRARY_PATH
export PYTHONPATH=/work/hdd/bdne/aacharya2/ltu-ili:$PYTHONPATH

# Run the bridge script
python moment_activelearning.py
