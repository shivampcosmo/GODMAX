#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=kkgggk_1000
#SBATCH -p gpu
#SBATCH -C a100-80gb
#SBATCH --mem=256G
#SBATCH --gpus=4
#SBATCH --output=/mnt/ceph/users/spandey/paper_pge/GODMAX/run_scripts/pge/logs/%x.%j.out
#SBATCH --error=/mnt/ceph/users/spandey/paper_pge/GODMAX/run_scripts/pge/logs/%x.%j.err


module purge
module load python
source ~/miniconda3/bin/activate ili-sbi

cd /mnt/ceph/users/spandey/paper_pge/GODMAX/run_scripts/pge/
time srun --export=ALL python sample_params_v5_halofit.py --probes="kk,gg,gk" --lmax=1000 --num_warmup=2000 --num_samples=2000 --num_chains=16 --max_tree_depth=4 --bao_prior=False --model_matter="halofit"
echo "done"
