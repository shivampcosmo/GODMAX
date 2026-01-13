#!/bin/bash

# Script to delete all files/folders starting with fastpm_B2 in subfolders 0-1999
# Base directory
BASE_DIR="/mnt/ceph/users/spandey/fastpm-shivam/LH_HR"

# Counter for tracking progress
count=0
deleted_count=0

echo "Starting deletion of fastpm_B2* files in $BASE_DIR"
echo "=================================================="

# Loop through folders 0 to 1999
for i in $(seq 0 1999); do
    folder="$BASE_DIR/$i"

    # Check if folder exists
    if [ -d "$folder" ]; then
        # Check if there are any fastpm_B2* files/folders
        if ls "$folder"/fastpm_B2* 1> /dev/null 2>&1; then
            echo "Processing folder $i..."
            # Delete all files and folders starting with fastpm_B2
            rm -rf "$folder"/fastpm_B2*
            # echo "Would delete: $folder/fastpm_B2*"
            deleted_count=$((deleted_count + 1))
        fi
        count=$((count + 1))
    else
        echo "Warning: Folder $folder does not exist, skipping..."
    fi

    # Print progress every 100 folders
    if [ $((i % 100)) -eq 0 ]; then
        echo "Progress: Processed $count folders, deleted from $deleted_count folders"
    fi
done

echo "=================================================="
echo "Done! Processed $count folders total."
echo "Deleted fastpm_B2* files from $deleted_count folders."
