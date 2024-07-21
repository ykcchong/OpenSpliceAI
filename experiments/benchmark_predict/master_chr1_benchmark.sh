#!/bin/bash

# Master benchmark script: predict on chr1
# Params: flanking size, model type (keras or pytorch)

# Check for correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model_type> <flanking_size>"
    exit 1
fi

MODEL_TYPE=$1
FLANKING_SIZE=$2

# Set paths
cd /ccb/cybertron/smao10/openspliceai

NATIVE_DIR="./experiments/benchmark_predict"
SETUP_SCRIPT="setup.py"
DATA_PATH="./data/toy/human/chr1.fa"
ANNOTATION_PATH="./data/toy/human/chr1_subset500.gff"
THRESHOLD=0.9

# Run the setup script
python $SETUP_SCRIPT install

USE_ANNOTATION=1

# Set the model path based on model type
if [ "$MODEL_TYPE" == "pytorch" ]; then
    MODEL_PATH="./models/spliceai-mane/${FLANKING_SIZE}nt/model_${FLANKING_SIZE}nt_rs42.pt"
    SCRIPT_PATH="./experiments/benchmark_predict/predict_test.py"
    OUTPUT_PARENT_DIR="$NATIVE_DIR/results/pytorch_chr1_${FLANKING_SIZE}nt_anno${USE_ANNOTATION}"
elif [ "$MODEL_TYPE" == "keras" ]; then
    SCRIPT_PATH="./experiments/benchmark_predict/spliceai_default_test.py"
    OUTPUT_PARENT_DIR="$NATIVE_DIR/results/keras_chr1_${FLANKING_SIZE}nt_anno${USE_ANNOTATION}"
else
    echo "Invalid model type. Please choose 'keras' or 'pytorch'."
    exit 1
fi

# Create the parent directory for this set of trials
mkdir -p "$OUTPUT_PARENT_DIR"

# Run the benchmarking and Scalene profiling 5 times
for i in {1..5}; do
    OUTPUT_PATH="$OUTPUT_PARENT_DIR/trial_$i"
    OUTPUT_FILE="$OUTPUT_PATH/output.log"
    ERROR_FILE="$OUTPUT_PATH/error.log"
    SCALENE_COMMAND="scalene --outfile $OUTPUT_PATH/scalene.html"

    mkdir -p "$OUTPUT_PATH"

    if [ "$USE_ANNOTATION" -eq 1 ]; then
        # with annotation
        COMMAND="$SCALENE_COMMAND --- $SCRIPT_PATH -m $MODEL_PATH -o $OUTPUT_PATH -f $FLANKING_SIZE -i $DATA_PATH -a $ANNOTATION_PATH -t $THRESHOLD > $OUTPUT_FILE 2> $ERROR_FILE"
    else
        # without annotation
        COMMAND="$SCALENE_COMMAND --- $SCRIPT_PATH -m $MODEL_PATH -o $OUTPUT_PATH -f $FLANKING_SIZE -i $DATA_PATH -t $THRESHOLD > $OUTPUT_FILE 2> $ERROR_FILE"
    fi

    # Echo the command to verify it
    echo "Running trial $i for $MODEL_TYPE model with flanking size $FLANKING_SIZE..."
    echo $COMMAND

    # Execute the command
    eval $COMMAND
done
