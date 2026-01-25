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

# Run the bridge script
python3 moment_al.py
