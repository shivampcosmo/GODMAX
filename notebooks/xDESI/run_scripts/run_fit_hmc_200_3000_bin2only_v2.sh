#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --job-name=fit_hmc_200_3000_800_2400_bin2only_v2
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
time srun python run_fit_abacus_test.py 200 3000 800 2400 2 '[2]' 200;
echo "done";

