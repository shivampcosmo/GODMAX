#!/bin/bash
#SBATCH --job-name=hmc_vs_sbi_2pt
#SBATCH --output=hmc_vs_sbi_2pt_%j.log
#SBATCH --error=hmc_vs_sbi_2pt_%j.err
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --time=00:20:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate godmax_env

# ── JAX memory: don't pre-allocate everything so PyTorch SBI can also use GPU
# ── if needed; SBI training runs on CPU in this script so this is a safety net
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# ── PyTorch: allow it to find the GPU but SBI training is pinned to CPU
export CUDA_VISIBLE_DEVICES=0

# ── NumPyro: set number of host devices for chain parallelism
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

cd /work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/sbi_Cls

python3 run_hmc_vs_sbi_2pt_noisy.py \
    --hmc-num-warmup  8000    \
    --hmc-num-samples 8000    \
    --hmc-num-chains  4       \
    --sbi-n-samples   8000    \
    --probes gy,gtau,gkappa,all_2pt \
    --output-dir /work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/sbi_Cls/hmc_vs_sbi_outputs
echo "=============================="
echo "End time : $(date)"
echo "=============================="
