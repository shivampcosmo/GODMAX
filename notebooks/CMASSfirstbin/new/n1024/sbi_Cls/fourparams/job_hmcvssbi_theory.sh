#!/bin/bash
#SBATCH --job-name=thhmc_sbi
#SBATCH --output=th_hmcvssbi_2pt_%j.log
#SBATCH --error=th_hmcvssbi_2pt_%j.err
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --time=00:30:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate godmax_env

# JAX: don't pre-allocate everything so PyTorch SBI can also use GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# PyTorch: allow it to find the GPU; SBI training uses device="auto"
export CUDA_VISIBLE_DEVICES=0

# NumPyro: set number of host devices for chain parallelism (4 HMC chains)
export XLA_FLAGS="--xla_force_host_platform_device_count=4"

echo "=============================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "CPUs       : $SLURM_CPUS_PER_TASK"
echo "Start time : $(date)"
echo "=============================="

echo "Python     : $(which python3)"
python3 -c "
import torch
print('PyTorch   :', torch.__version__, '| CUDA:', torch.cuda.is_available(),
      '| Devices:', torch.cuda.device_count())
import jax
print('JAX       :', jax.__version__, '| Devices:', jax.devices())
"

cd /work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/sbi_Cls/fourparams

python3 run_hmcvssbi_theory.py

echo "=============================="
echo "End time : $(date)"
echo "=============================="
