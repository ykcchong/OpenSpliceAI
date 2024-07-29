#!/bin/bash

# Master benchmark script: predict on random genes from MANE
# Params: flanking size, model type (keras or pytorch), subset size

## args: variant_test.py [-h] -R reference -A annotation [-I [input]] [-O [output]] [-D [distance]] [-M [mask]] [--model MODEL] [--flanking-size FLANKING_SIZE] [--model-type {keras,pytorch}] [--precision PRECISION]
## variant_test.py: error: the following arguments are required: -R, -A

# Check for correct number of arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <model_type> <flanking_size> <subset_size>"
    exit 1
fi

MODEL_TYPE=$1
FLANKING_SIZE=$2
SUBSET_SIZE=$3

# Set paths
NATIVE_DIR="./experiments/benchmark_variant"
DATA_PATH="./data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna"
VCF_PATH="./data/vcf/hg38_mills100g_subset${SUBSET_SIZE}.vcf"
ANNOTATION_PATH="./data/vcf/grch38.txt"

# Set the script path and output parent directory based on model type
if [ "$MODEL_TYPE" == "pytorch" ]; then
    MODEL_PATH="./models/spliceai-mane/${FLANKING_SIZE}nt/model_${FLANKING_SIZE}nt_rs42.pt"
    SCRIPT_PATH="./experiments/benchmark_variant/variant_test.py"
    OUTPUT_PARENT_DIR="$NATIVE_DIR/results_${SUBSET_SIZE}/pytorch_VCF_sub${SUBSET_SIZE}_${FLANKING_SIZE}nt"
    MODEL_ARG="-m $MODEL_PATH"
elif [ "$MODEL_TYPE" == "keras" ]; then
    SCRIPT_PATH="./experiments/benchmark_variant/spliceai_orig.py"
    OUTPUT_PARENT_DIR="$NATIVE_DIR/results_${SUBSET_SIZE}/keras_VCF_sub${SUBSET_SIZE}_${FLANKING_SIZE}nt"
    MODEL_ARG=""
else
    echo "Invalid model type. Please choose 'keras' or 'pytorch'."
    exit 1
fi

# Create the parent directory for this set of trials
mkdir -p "$OUTPUT_PARENT_DIR"

# Run the benchmarking and Scalene profiling 5 times
for i in {1..5}; do
    OUTPUT_PATH="$OUTPUT_PARENT_DIR/trial_$i/"
    OUTPUT_FILE="$OUTPUT_PATH/output.log"
    ERROR_FILE="$OUTPUT_PATH/error.log"
    OUTPUT_VCF="$OUTPUT_PATH/result.vcf"
    SCALENE_COMMAND="scalene --outfile $OUTPUT_PATH/scalene.html --no-browser"

    mkdir -p "$OUTPUT_PATH"

    COMMAND="$SCALENE_COMMAND --- $SCRIPT_PATH $MODEL_ARG -I $VCF_PATH -O $OUTPUT_VCF -f $FLANKING_SIZE -R $DATA_PATH -A $ANNOTATION_PATH > $OUTPUT_FILE 2> $ERROR_FILE"

    # Echo the command to verify it
    echo "Running trial $i for $MODEL_TYPE model with flanking size $FLANKING_SIZE..."
    echo $COMMAND

    # Execute the command
    eval $COMMAND
done
