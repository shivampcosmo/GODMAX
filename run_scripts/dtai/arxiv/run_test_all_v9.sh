#!/bin/bash
#SBATCH --account=bdne-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=15
#SBATCH --time=16:00:00
#SBATCH --partition=ghx4
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --job-name=v9_ALL_run
#SBATCH --output=/projects/bdne/spandey3/new_godmax/GODMAX/run_scripts/dtai/logs/%x.%j.out
#SBATCH --error=/projects/bdne/spandey3/new_godmax/GODMAX/run_scripts/dtai/logs/%x.%j.err
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JAX_TRACEBACK_FILTERING=off


module purge
module load python
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/sw/user/python/miniforge3-pytorch-2.5.0/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh" ]; then
        . "/sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh"
    else
        export PATH="/sw/user/python/miniforge3-pytorch-2.5.0/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate /u/spandey3/myjax
which python
module load cuda

which python
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
cd /projects/bdne/spandey3/new_godmax/GODMAX/run_scripts/dtai/
time srun --export=ALL python sample_params_v9.py cib_1p7_dBeta all DMB 0 poweradd
echo "done"
