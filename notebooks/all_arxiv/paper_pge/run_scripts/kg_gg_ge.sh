#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --job-name=gg_gk_ge
#SBATCH -p gpu
#SBATCH -C a100-80gb
#SBATCH --gpus=1
#SBATCH --output=/mnt/home/spandey/ceph/GODMAX/notebooks/paper_pge/run_scripts/logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/GODMAX/notebooks/paper_pge/run_scripts/logs/%x.%j.err

source ~/.bashrc
conda activate ili-sbi

cd /mnt/home/spandey/ceph/GODMAX/notebooks/paper_pge/

time srun python get_fisher_mat.py "gg,gk,ge" 500
time srun python get_fisher_mat.py "gg,gk,ge" 1000
time srun python get_fisher_mat.py "gg,gk,ge" 2000
time srun python get_fisher_mat.py "gg,gk,ge" 4000
time srun python get_fisher_mat.py "gg,gk,ge" 8000
echo "done"
