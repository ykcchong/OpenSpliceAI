#!/bin/bash
#SBATCH -N 1
#SBATCH --time=72:0:0
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH -A ssalzbe1_gpu
#SBATCH --array=0-39
#SBATCH --mem=64G  # Request 64 GB of memory

# Define the flanking sizes
FLANKING_SIZES=(80 400 2000 10000)
# FLANKING_SIZES=(80)

# Calculate the total number of experiments per combination of flanking size
NUM_EXPERIMENTS=10

# Calculate the current flanking size, experiment number, and random seed based on the array task ID
FLANKING_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_EXPERIMENTS))
EXP_NUM=$((SLURM_ARRAY_TASK_ID % NUM_EXPERIMENTS))
RANDOM_SEED=$((10 + EXP_NUM))  # Adjust the random seed as needed

FLANKING_SIZE=${FLANKING_SIZES[$FLANKING_INDEX]}
echo "Flanking size: $FLANKING_SIZE"
echo "Experiment number: $EXP_NUM"

python keras_mutagenesis.py /home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/experiments/mutagenesis/experiment_2 ${EXP_NUM} $FLANKING_SIZE