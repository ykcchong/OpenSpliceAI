#!/bin/bash

# example from spliceai repo
# usage: openspliceai variant [-h] [--model MODEL] [--flanking-size FLANKING_SIZE] [-I [input]] [-O [output]] -R reference -A annotation [-D [distance]] [-M [mask]]
# openspliceai variant: error: the following arguments are required: -R, -A
# NOTE: calls predict method, so will need the same arguments generally

SETUP="/ccb/cybertron/smao10/openspliceai/setup.py"
RESULT_DIR="./results/variant/default"

# Run the setup script
python "$SETUP" install
if [ -d "$RESULT_DIR" ]; then
    rm -rf "$RESULT_DIR"
fi
mkdir -p $RESULT_DIR

REF_GENOME_PATH="/ccb/cybertron/smao10/openspliceai/data/ref_genome/homo_sapiens/GRCh37/hg19.fa"
ANNOTATION_PATH="grch37" # NOTE: this is a custom annotation file
MODEL_PATH="SpliceAI"
INPUT_PATH="./data/vcf/input.vcf"
OUTPUT_PATH="$RESULT_DIR/output.vcf"
FLANKING_SIZE=5000
MODEL_TYPE="keras"

OUTPUT_FILE="$RESULT_DIR/output.log"
ERROR_FILE="$RESULT_DIR/error.log"

# Run the openspliceai variant command
echo openspliceai variant -R "$REF_GENOME_PATH" -A "$ANNOTATION_PATH" -m "$MODEL_PATH" -f $FLANKING_SIZE -I "$INPUT_PATH" -O "$OUTPUT_PATH" \
> "$OUTPUT_FILE" 2> "$ERROR_FILE"
openspliceai variant -R "$REF_GENOME_PATH" -A "$ANNOTATION_PATH" -m "$MODEL_PATH" -f $FLANKING_SIZE -I "$INPUT_PATH" -O "$OUTPUT_PATH" \
> "$OUTPUT_FILE" 2> "$ERROR_FILE"