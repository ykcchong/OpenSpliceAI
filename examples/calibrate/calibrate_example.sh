#!/bin/bash
# A script to run the openspliceai create-data command with the required arguments

# Resolve the parent directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "$PDIR"

CURRENT_SPECIES="MANE"
FLANKING_SIZE=10000
RANDOM_SEED=11

# Set up output directory
OUTPUT_DIR="$PDIR/examples/calibrate/results/"
mkdir -p $OUTPUT_DIR
OUTPUT_FILE="$OUTPUT_DIR/output.log"
ERROR_FILE="$OUTPUT_DIR/error.log"

# Set up dataset paths
TRAIN_DATASET="$PDIR/examples/create-data/results/dataset_train.h5"
TEST_DATASET="$PDIR/examples/create-data/results/dataset_test.h5"
CURRENT_SPECIES_lowercase="${CURRENT_SPECIES,,}"

PRETRAINED_MODEL="$PDIR/examples/train/results/model.pt"

echo "Current species: $CURRENT_SPECIES"
echo "Flanking size: $FLANKING_SIZE"
echo "Random seed: $RANDOM_SEED"
echo "Output directory: $OUTPUT_DIR"
echo "Train dataset: $TRAIN_DATASET"
echo "Test dataset: $TEST_DATASET"
echo "Pretrained model: $PRETRAINED_MODEL"

# Run the OpenSpliceAI calibrate command
echo openspliceai calibrate --flanking-size $FLANKING_SIZE \
    --train-dataset $TRAIN_DATASET \
    --test-dataset $TEST_DATASET \
    --output-dir $OUTPUT_DIR \
    --project-name human_${CURRENT_SPECIES}_calibrate \
    --random-seed ${RANDOM_SEED} \
    --pretrained-model $PRETRAINED_MODEL \
    --loss cross_entropy_loss > $OUTPUT_FILE 2> $ERROR_FILE
