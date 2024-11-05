#!/bin/bash

# Predict on the complete Mills/1000G dataset from MANE using both Keras and PyTorch models
# RUNS ONCE FOR BOTH KERAS AND PYTORCH MODELS (10k)

# Set the flanking size
FLANKING_SIZE=10000

# Set paths
NATIVE_DIR="/ccb/cybertron/smao10/openspliceai/experiments/benchmark_variant"
DATA_PATH="/ccb/cybertron/smao10/openspliceai/data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna"
VCF_PATH="/ccb/cybertron/smao10/openspliceai/data/vcf/hg38_mills1000g_subset25000.vcf"
ANNOTATION_PATH="/ccb/cybertron/smao10/openspliceai/data/vcf/grch38.txt"
OUTPUT_PARENT_DIR="/ccb/cybertron/smao10/openspliceai/experiments/benchmark_variant/comparison"

# Create the parent directory for comparison results
mkdir -p "$OUTPUT_PARENT_DIR"

# Run benchmark once for both keras and pytorch
# for MODEL_TYPE in "pytorch" "keras"; do
for MODEL_TYPE in "keras"; do
    if [ "$MODEL_TYPE" == "pytorch" ]; then
        MODEL_PATH="/ccb/cybertron/smao10/openspliceai/models/spliceai-mane/${FLANKING_SIZE}nt/model_${FLANKING_SIZE}nt_rs14.pth"
        SCRIPT_PATH="/ccb/cybertron/smao10/openspliceai/experiments/benchmark_variant/variant_test.py"
        MODEL_ARG="-m $MODEL_PATH"
    elif [ "$MODEL_TYPE" == "keras" ]; then
        SCRIPT_PATH="/ccb/cybertron/smao10/openspliceai/experiments/benchmark_variant/spliceai_orig.py"
        MODEL_ARG=""
    else
        echo "Invalid model type. Please choose 'keras' or 'pytorch'."
        exit 1
    fi

    # Define output paths for this model type
    OUTPUT_PATH="$OUTPUT_PARENT_DIR/${MODEL_TYPE}/"
    OUTPUT_FILE="$OUTPUT_PATH/output.log"
    ERROR_FILE="$OUTPUT_PATH/error.log"
    OUTPUT_VCF="$OUTPUT_PATH/result.vcf"

    # Create the output directory for this model type
    mkdir -p "$OUTPUT_PATH"

    # Construct the command
    COMMAND="python3 $SCRIPT_PATH $MODEL_ARG -I $VCF_PATH -O $OUTPUT_VCF --flanking-size $FLANKING_SIZE --precision 6 -R $DATA_PATH -A $ANNOTATION_PATH > $OUTPUT_FILE 2> $ERROR_FILE"

    # Echo the command to verify it
    echo "Running benchmark for $MODEL_TYPE model with flanking size $FLANKING_SIZE..."
    echo $COMMAND

    # Execute the command
    eval $COMMAND
done