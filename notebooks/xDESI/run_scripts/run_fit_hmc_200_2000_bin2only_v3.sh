#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name=fit_hmc_200_2000_400_800_bin2only_v3
#SBATCH -p gpu
#SBATCH -C a100-80gb
#SBATCH --gpus=4
#SBATCH --output=/mnt/ceph/users/spandey/paste_godmax/GODMAX/notebooks/xDESI/run_scripts/slurm_scripts/logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/paste_godmax/GODMAX/notebooks/xDESI/run_scripts/slurm_scripts/logs/%x.%j.err

source /etc/profile.d/modules.sh
module purge
module load openmpi/4.1.8
module load python
source ~/miniconda3/bin/activate ili-sbi

cd "/mnt/ceph/users/spandey/paste_godmax/GODMAX/notebooks/xDESI";
echo "$PWD";
time srun python run_fit_abacus_test.py 200 2000 400 800 3 '[2]';
echo "done";

