#!/bin/bash

# Script to run master benchmark script for all model types and flanking sizes

# Define model types and flanking sizes
MODEL_TYPES=("pytorch" "keras")
FLANKING_SIZES=(80 400 2000 10000)
SUBSET_SIZE=$1 # You can change this value as needed

# Set paths
cd /home/smao10/OpenSpliceAI

NATIVE_DIR="./experiments/benchmark_predict"
SETUP_SCRIPT="setup.py"

# Run the setup script
python $SETUP_SCRIPT install

# Iterate over each model type and flanking size
for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
    for FLANKING_SIZE in "${FLANKING_SIZES[@]}"; do
        echo "Running benchmark for model type: $MODEL_TYPE with flanking size: $FLANKING_SIZE and subset size: $SUBSET_SIZE"
        ./experiments/benchmark_predict/master_MANE_benchmark.sh $MODEL_TYPE $FLANKING_SIZE $SUBSET_SIZE
    done
done
