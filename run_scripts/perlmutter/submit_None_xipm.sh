#!/bin/bash
#SBATCH -A des
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-task=4
#SBATCH --gpu-bind=none
#SBATCH --job-name=NUTS2_UP_run_deproj_cib_1p7_dBeta_probe_xipm
#SBATCH --output=/global/cfs/cdirs/lsst/www/shivamp/GODMAX/run_scripts/perlmutter/logs/%x.%j.out
#SBATCH --error=/global/cfs/cdirs/lsst/www/shivamp/GODMAX/run_scripts/perlmutter/logs/%x.%j.err


export SLURM_CPU_BIND="cores"

# module purge
module load python
module load conda
module load texlive
module load cudatoolkit
module load cudnn
conda activate /global/cfs/cdirs/lsst/www/shivamp/env/jax_godmax


cd /global/cfs/cdirs/lsst/www/shivamp/GODMAX/run_scripts/perlmutter/

time srun python sample_params.py cib_1p7_dBeta xip_xim
echo "done"
