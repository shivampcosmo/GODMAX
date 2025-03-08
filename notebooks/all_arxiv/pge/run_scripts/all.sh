#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --job-name=all_probes
#SBATCH -p gpu
#SBATCH -C a100-80gb
#SBATCH --gpus=1
#SBATCH --output=/mnt/home/spandey/ceph/GODMAX/notebooks/pge/run_scripts/logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/GODMAX/notebooks/pge/run_scripts/logs/%x.%j.err

source ~/.bashrc
conda activate ili-sbi

cd /mnt/home/spandey/ceph/GODMAX/notebooks/pge/

time srun python get_fisher_wpge.py "ky,kk,gg,gy,gk,ge" 500
time srun python get_fisher_wpge.py "ky,kk,gg,gy,gk,ge" 1000
time srun python get_fisher_wpge.py "ky,kk,gg,gy,gk,ge" 2000
time srun python get_fisher_wpge.py "ky,kk,gg,gy,gk,ge" 4000
time srun python get_fisher_wpge.py "ky,kk,gg,gy,gk,ge" 8000
echo "done"
