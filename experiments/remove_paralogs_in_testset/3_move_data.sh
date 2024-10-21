#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=72:0:0
#SBATCH --partition=parallel
#SBATCH -A ssalzbe1_gpu
#SBATCH --mem=16G
#SBATCH --array=0-4

# Define an array of species
SPECIES=(MANE mouse zebrafish honeybee arabidopsis)

# Get the species corresponding to the SLURM array index
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

mv /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_${species}/dataset_test.h5 /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_${species}/dataset_test.h5.bak

cp ${species}_data/filtered_test_dataset.h5 /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_${species}/dataset_test.h5
