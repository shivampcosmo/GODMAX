#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=2:40:00
#SBATCH --job-name=run_5_8_2048
#SBATCH -p gpu
#SBATCH -C a100-80gb
#SBATCH --mem=384G
#SBATCH --gpus=1
#SBATCH --output=/mnt/home/spandey/ceph/paste_godmax/GODMAX/notebooks/pasting/run_scripts/logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/paste_godmax/GODMAX/notebooks/pasting/run_scripts/logs/%x.%j.err

# module purge

module purge
module load python
source ~/miniconda3/bin/activate ili-sbi

cd /mnt/home/spandey/ceph/paste_godmax/GODMAX/notebooks/pasting/
time srun python run_nside_map_split_halfdome.py 2048 5 8
echo "done"
