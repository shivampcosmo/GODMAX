#!/bin/bash
#SBATCH -A des
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-task=4
#SBATCH --gpu-bind=none
#SBATCH --job-name=FINAL1_run_deproj_cib_1p7_dBeta_probe_all_sample_thetaco
#SBATCH --output=/global/cfs/cdirs/lsst/www/shivamp/GODMAX/run_scripts/perlmutter/logs/%x.%j.out
#SBATCH --error=/global/cfs/cdirs/lsst/www/shivamp/GODMAX/run_scripts/perlmutter/logs/%x.%j.err


export SLURM_CPU_BIND="cores"

# module purge
module load cudatoolkit
module load cudnn
module load python
module load conda
module load texlive
conda activate /global/cfs/cdirs/lsst/www/shivamp/env/jax_godmax2


cd /global/cfs/cdirs/lsst/www/shivamp/GODMAX/run_scripts/perlmutter/

time srun python sample_params_thetaco.py cib_1p7_dBeta all
echo "done"