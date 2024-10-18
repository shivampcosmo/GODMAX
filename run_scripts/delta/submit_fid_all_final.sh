#!/bin/bash
#SBATCH --account=bdne-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --gpus-per-node=4                  # total number of GPUs
#SBATCH --cpus-per-task=10         # CPU cores per MPI process
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --mem=80G          # host memory per CPU core
#SBATCH --time=1-12:00            # time (DD-HH:MM)
#SBATCH --job-name=test_fid_all_final
#SBATCH --output=/projects/bdne/spandey3/GODMAX/run_scripts/delta/logs/%x.%j.out
#SBATCH --error=/projects/bdne/spandey3/GODMAX/run_scripts/delta/logs/%x.%j.err
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JAX_TRACEBACK_FILTERING=off

# export PATH="$HOME/.local/bin:$HOME/bin:$PATH"
module load python
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/sw/external/python/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sw/external/python/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/sw/external/python/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/sw/external/python/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
module load cuda/12.3.0
conda activate /projects/bdne/spandey3/envs/charm
which python
srun --export=ALL python /projects/bdne/spandey3/GODMAX/run_scripts/delta/sample_params_simple.py cib_1p7_dBeta all
echo "done"
