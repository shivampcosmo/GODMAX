#!/bin/bash
#SBATCH --job-name=sbi_cls
#SBATCH --output=sbicls_%j.log
#SBATCH --error=sbicls_%j.err
#SBATCH --partition=ghx4
#SBATCH --account=bdne-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --time=06:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate godmax_env

echo "Python:  $(which python3)"
python3 -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

#python3 sbi_and_activelearning.py
python3 sbi_on_Cls.py
