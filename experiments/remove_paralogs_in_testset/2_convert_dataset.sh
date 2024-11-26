#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=72:0:0
#SBATCH --partition=parallel
#SBATCH -A ssalzbe1_gpu
#SBATCH --mem=64G
#SBATCH --array=0-4

# Define an array of species
SPECIES=(MANE mouse zebrafish honeybee arabidopsis)

# Get the species corresponding to the SLURM array index
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

echo "Running job for species: $species"
python 2_convert_dataset.py \
    --input_file ${species}_data/filtered_test_data.h5 \
    --output_file ${species}_data/filtered_test_dataset.h5
