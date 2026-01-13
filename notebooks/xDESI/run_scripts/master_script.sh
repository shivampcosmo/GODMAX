#!/bin/bash

# --- Configuration ---
# Set the parameters for your runs
TOTAL_DEVICES=8

# Set the base directory
WORK_DIR="/mnt/ceph/users/spandey/paste_godmax/GODMAX/notebooks/xDESI/run_scripts"

# Set the directory to store the generated Slurm scripts and logs
SCRIPT_DIR="${WORK_DIR}/slurm_scripts"
LOG_DIR="${SCRIPT_DIR}/slurm_scripts/logs"

# --- Script Logic ---

# Create the directories if they don't exist
mkdir -p "${SCRIPT_DIR}"
mkdir -p "${LOG_DIR}"

echo "Configuration:"
echo "TOTAL_DEVICES:  ${TOTAL_DEVICES}"
echo "Working Dir:    ${WORK_DIR}"
echo "Script Dir:     ${SCRIPT_DIR}"
echo "Log Dir:        ${LOG_DIR}"
echo "--------------------------------"

# Loop from 0 to TOTAL_DEVICES - 1
for (( JDEVICE=0; JDEVICE<TOTAL_DEVICES; JDEVICE++ )); do
    # Calculate i range for this device
    I_START=$((2 * JDEVICE))
    I_END=$((2 * (JDEVICE + 1)))

    # Define unique names for the job and the script file
    JOB_NAME="measure_${JDEVICE}_${I_START}_${I_END}"
    SLURM_SCRIPT_PATH="${SCRIPT_DIR}/${JOB_NAME}.slurm"

    echo "Generating script: ${SLURM_SCRIPT_PATH} (i: ${I_START}..${I_END})"

    # Use a "Here Document" to write the Slurm script
    cat > "${SLURM_SCRIPT_PATH}" <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH -C rome
#SBATCH -p cmbas
#SBATCH --time=01:00:00
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/%x.%j.out
#SBATCH --error=${LOG_DIR}/%x.%j.err

echo "--- JOB DETAILS ---"
echo "Job Name: \${SLURM_JOB_NAME} (\${SLURM_JOB_ID})"
echo "Running on: \$(hostname)"
echo "JDEVICE: ${JDEVICE}"
echo "I_START: ${I_START}"
echo "I_END: ${I_END}"
echo "-------------------"

module purge
module load openmpi/4.1.8
module load python
source ~/miniconda3/bin/activate ili-sbi

# Loop through i values for this device
for i in {${I_START}..$((I_END - 1))};
do
    echo \$i;
    cd "/mnt/ceph/users/spandey/paste_godmax/GODMAX/notebooks/xDESI";
    echo "\$PWD";
    time srun python measure_abacus_Clgg.py \$i;
    echo "done";
done

echo "All runs complete for device ${JDEVICE}"
EOF

    # Submit the generated script to Slurm
    sbatch "${SLURM_SCRIPT_PATH}"

done

echo "--------------------------------"
echo "All ${TOTAL_DEVICES} jobs have been submitted."