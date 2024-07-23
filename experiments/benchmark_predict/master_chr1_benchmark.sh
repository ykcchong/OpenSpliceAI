#!/bin/bash

# Master benchmark script: predict on chr1
# Params: flanking size, model type (keras or pytorch), subset size

# Check for correct number of arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <model_type> <flanking_size> <subset_size>"
    exit 1
fi

MODEL_TYPE=$1
FLANKING_SIZE=$2
SUBSET_SIZE=$3

# Set paths
NATIVE_DIR="./experiments/benchmark_predict"
DATA_PATH="./data/toy/human/chr1.fa"
ANNOTATION_PATH="./data/toy/human/chr1_subset${SUBSET_SIZE}.gff"
THRESHOLD=0.9

USE_ANNOTATION=1

# Set the script path and output parent directory based on model type
if [ "$MODEL_TYPE" == "pytorch" ]; then
    MODEL_PATH="./models/spliceai-mane/${FLANKING_SIZE}nt/model_${FLANKING_SIZE}nt_rs42.pt"
    SCRIPT_PATH="./experiments/benchmark_predict/predict_test.py"
    OUTPUT_PARENT_DIR="$NATIVE_DIR/results/pytorch_chr1_sub${SUBSET_SIZE}_${FLANKING_SIZE}nt_anno${USE_ANNOTATION}"
    MODEL_ARG="-m $MODEL_PATH"
elif [ "$MODEL_TYPE" == "keras" ]; then
    SCRIPT_PATH="./experiments/benchmark_predict/spliceai_default_test.py"
    OUTPUT_PARENT_DIR="$NATIVE_DIR/results/keras_chr1_sub${SUBSET_SIZE}_${FLANKING_SIZE}nt_anno${USE_ANNOTATION}"
    MODEL_ARG=""
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
    SCALENE_COMMAND="scalene --outfile $OUTPUT_PATH/scalene.html --no-browser"

    mkdir -p "$OUTPUT_PATH"

    if [ "$USE_ANNOTATION" -eq 1 ]; then
        # with annotation
        COMMAND="$SCALENE_COMMAND --- $SCRIPT_PATH $MODEL_ARG -o $OUTPUT_PATH -f $FLANKING_SIZE -i $DATA_PATH -a $ANNOTATION_PATH -t $THRESHOLD > $OUTPUT_FILE 2> $ERROR_FILE"
    else
        # without annotation
        COMMAND="$SCALENE_COMMAND --- $SCRIPT_PATH $MODEL_ARG -o $OUTPUT_PATH -f $FLANKING_SIZE -i $DATA_PATH -t $THRESHOLD > $OUTPUT_FILE 2> $ERROR_FILE"
    fi

    # Echo the command to verify it
    echo "Running trial $i for $MODEL_TYPE model with flanking size $FLANKING_SIZE..."
    echo $COMMAND

    # Execute the command
    eval $COMMAND

    # Remove the output directory except for the Scalene output
    rm -r "$OUTPUT_PATH/SpliceAI_5000_${FLANKING_SIZE}"
done
