#!/bin/bash

SETUP="/ccb/cybertron/smao10/openspliceai/setup.py"
RESULT_DIR="./results/variant/pytorch"

# Run the setup script
python "$SETUP" install
if [ -d "$RESULT_DIR" ]; then
    rm -rf "$RESULT_DIR"
fi
mkdir -p $RESULT_DIR

REF_GENOME_PATH="/ccb/cybertron/smao10/openspliceai/data/ref_genome/homo_sapiens/GRCh37/hg19.fa"
ANNOTATION_PATH="grch37" # NOTE: this is a custom annotation file
MODEL_PATH="./models/spliceai-mane/400nt/model_400nt_rs40.pt"
INPUT_PATH="./data/vcf/input.vcf"
OUTPUT_PATH="$RESULT_DIR/output.vcf"
FLANKING_SIZE=400
MODEL_TYPE="pytorch"

OUTPUT_FILE="$RESULT_DIR/output.log"
ERROR_FILE="$RESULT_DIR/error.log"

# Run the spliceai-toolkit variant command
echo spliceai-toolkit variant -R "$REF_GENOME_PATH" -A "$ANNOTATION_PATH" -m "$MODEL_PATH" -f $FLANKING_SIZE -t "$MODEL_TYPE" -I "$INPUT_PATH" -O "$OUTPUT_PATH" \
> "$OUTPUT_FILE" 2> "$ERROR_FILE"
spliceai-toolkit variant -R "$REF_GENOME_PATH" -A "$ANNOTATION_PATH" -m "$MODEL_PATH" -f $FLANKING_SIZE -t "$MODEL_TYPE" -I "$INPUT_PATH" -O "$OUTPUT_PATH" \
> "$OUTPUT_FILE" 2> "$ERROR_FILE"