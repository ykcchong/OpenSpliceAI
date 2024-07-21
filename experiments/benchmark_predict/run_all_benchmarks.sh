#!/bin/bash

# Script to run master benchmark script for all model types and flanking sizes

# Define model types and flanking sizes
MODEL_TYPES=("pytorch" "keras")
FLANKING_SIZES=(80 400 2000 10000)

# Iterate over each model type and flanking size
for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
    for FLANKING_SIZE in "${FLANKING_SIZES[@]}"; do
        echo "Running benchmark for model type: $MODEL_TYPE with flanking size: $FLANKING_SIZE"
        ./master_benchmark.sh $MODEL_TYPE $FLANKING_SIZE
    done
done
