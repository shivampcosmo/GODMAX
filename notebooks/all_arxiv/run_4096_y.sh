#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --job-name=test_4096
#SBATCH -p gpu
#SBATCH -C a100-80gb
#SBATCH --mem=768G
#SBATCH --gpus=1
#SBATCH --output=/mnt/home/spandey/ceph/GODMAX/run_scripts/FI/logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/GODMAX/run_scripts/FI/logs/%x.%j.err

# module purge

module purge
module load python
source ~/miniconda3/bin/activate ili-sbi

cd /mnt/home/spandey/ceph/GODMAX/notebooks/all_arxiv/
# time srun python run_nside_8192_ksz.py
time srun python run_nside_4096_tsz.py
echo "done"
