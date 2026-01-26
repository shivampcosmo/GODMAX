#!/bin/bash
#SBATCH --job-name=sbiili
#SBATCH --output=sbiili_%j.log
#SBATCH --error=sbiili_%j.err
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

# Run the bridge script
python3 sbi_and_activelearning.py
