#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=28:00:00
#SBATCH --job-name=NUTS2_UP_run_deproj_cib_1p7_dBeta_probe_xip+xim
#SBATCH -p gpu
#SBATCH -C a100-80gb,ib
#SBATCH --gpus=4
#SBATCH --output=/mnt/home/spandey/ceph/GODMAX/run_scripts/test_runs_v0/logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/GODMAX/run_scripts/test_runs_v0/logs/%x.%j.err

# module purge
module load python
module load gcc
module load cuda
module load cudnn
module load nccl
module load modules/2.2-20230808
source ~/miniconda3/bin/activate /mnt/home/spandey/venv_gm


cd /mnt/home/spandey/ceph/GODMAX/run_scripts/test_runs_v0/
time srun python sample_params.py cib_1p7_dBeta xip_xim
echo "done"