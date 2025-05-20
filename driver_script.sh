#!/bin/bash

# Print usage message
usage() {
    echo "Usage: $0 <path_to_root_folder> <bacteria_folder_name> <archaea_folder_name>"
    echo ""
    echo "Arguments:"
    echo "  <path_to_root_folder>    Absolute or relative path to the root data folder"
    echo "  <bacteria_folder_name>   Name of the folder with bacterial data (e.g., 'Bacteria Data')"
    echo "  <archaea_folder_name>    Name of the folder with archaeal data (e.g., 'Archaea Data')"
    exit 1
}

# Show help if needed
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

# Check number of arguments
if [ "$#" -ne 3 ]; then
    echo "Error: Incorrect number of arguments."
    usage
fi

# Actual script logic starts here
ROOT_PATH=$1  # C:/Users/gebeb/Desktop/Research/16S_rRNA/RM_nextflow/
BACTERIA_FOLDER=$2  # Nextflow_Ba
ARCHAEA_FOLDER=$3  # Nextflow_Ar

# Example: echo arguments
echo "Running with:"
echo "Root folder: $ROOT_FOLDER"
echo "Bacteria folder: $BACTERIA_FOLDER"
echo "Archaea folder: $ARCHAEA_FOLDER"

# This will save AVSs/taxa/metadata CSVs in each respective folder
python main.py $ROOT_PATH $BACTERIA_FOLDER $ARCHAEA_FOLDER
