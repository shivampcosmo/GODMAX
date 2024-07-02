#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=NUTS2_UP_run_deproj_cib_1p7_dBeta_probe_gty
#SBATCH -p gpu
#SBATCH -C a100-80gb,ib
#SBATCH --gpus=4
#SBATCH --output=/mnt/home/spandey/ceph/GODMAX/run_scripts/FI/logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/GODMAX/run_scripts/FI/logs/%x.%j.err

# module purge
module load python
module load gcc
module load cuda
module load cudnn
module load nccl
module load modules/2.2-20230808
source ~/miniconda3/bin/activate /mnt/home/spandey/venv_gm

cd /mnt/home/spandey/ceph/GODMAX/run_scripts/FI/

time srun python sample_params.py cib_1p7_dBeta gty
echo "done"
