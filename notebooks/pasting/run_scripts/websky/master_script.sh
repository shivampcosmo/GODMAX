#!/bin/bash

# --- Configuration ---
# Set the parameters for your runs
NSIDE=1024
TOTAL_DEVICES=4

# Set the base directory where your python script is located
# The script will `cd` into this directory before running python
WORK_DIR="/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/pasting/"

# Set the directory to store the generated Slurm scripts and logs
# Using a sub-directory keeps your project folder clean
SCRIPT_DIR="${WORK_DIR}/run_scripts/websky"
LOG_DIR="${SCRIPT_DIR}/logs"

# --- Script Logic ---

# Create the directories if they don't exist
# The -p flag ensures no error if the directory already exists
mkdir -p "${SCRIPT_DIR}"
mkdir -p "${LOG_DIR}"

echo "Configuration:"
echo "NSIDE:          ${NSIDE}"
echo "TOTAL_DEVICES:  ${TOTAL_DEVICES}"
echo "Working Dir:    ${WORK_DIR}"
echo "Script Dir:     ${SCRIPT_DIR}"
echo "Log Dir:        ${LOG_DIR}"
echo "--------------------------------"

# Loop from 0 to TOTAL_DEVICES - 1
for (( JDEVICE=0; JDEVICE<TOTAL_DEVICES; JDEVICE++ )); do
    # 1. Define unique names for the job and the script file
    JOB_NAME="run_${JDEVICE}_${TOTAL_DEVICES}_${NSIDE}"
    SLURM_SCRIPT_PATH="${SCRIPT_DIR}/${JOB_NAME}.slurm"

    echo "Generating script: ${SLURM_SCRIPT_PATH}"

    # 2. Use a "Here Document" to write the Slurm script
    # This is a clean way to write multi-line text files.
    # Variables like ${JOB_NAME}, ${LOG_DIR}, etc. will be automatically replaced.
    cat > "${SLURM_SCRIPT_PATH}" <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --job-name=${JOB_NAME}
#SBATCH -p gpu
#SBATCH -C a100-80gb
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --output=${LOG_DIR}/%x.%j.out
#SBATCH --error=${LOG_DIR}/%x.%j.err

echo "--- JOB DETAILS ---"
echo "Job Name: \${SLURM_JOB_NAME} (\${SLURM_JOB_ID})"
echo "Running on: \$(hostname)"
echo "NSIDE: ${NSIDE}"
echo "JDEVICE: ${JDEVICE}"
echo "TOTAL_DEVICES: ${TOTAL_DEVICES}"
echo "-------------------"

# Load necessary modules
module purge
module load python
source ~/miniconda3/bin/activate ili-sbi

# Change to the working directory
cd "${WORK_DIR}"

# Run the python script with the correct parameters
# The 'time' command will measure the execution duration
# time srun python run_nside_map_halfdome.py ${NSIDE} ${JDEVICE} ${TOTAL_DEVICES}
time srun python run_nside_tau_split_websky.py ${NSIDE} ${JDEVICE} ${TOTAL_DEVICES}

echo "done"
EOF

    # 3. Submit the generated script to Slurm
    sbatch "${SLURM_SCRIPT_PATH}"

done

echo "--------------------------------"
echo "All ${TOTAL_DEVICES} jobs have been submitted."