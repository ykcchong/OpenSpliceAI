#!/bin/bash

# run of grch38 with and without annotation

SETUP="/ccb/cybertron/smao10/openspliceai/setup.py"
RESULT_DIR="./results/predict/SpliceAI_5000_10000"

# Run the setup script
python "$SETUP" install
if [ -d "$RESULT_DIR" ]; then
    rm -rf "$RESULT_DIR"
fi
mkdir -p $RESULT_DIR

MODEL_PATH="./models/spliceai-mane/10000nt/model_10000nt_rs42.pt"
OUTPUT_PATH="./results/predict"
DATA_PATH="./data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna"
ANNOTATION_PATH="./data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.gff"
FLANKING_SIZE=10000
THRESHOLD=0.001

OUTPUT_FILE="$RESULT_DIR/output.log"
ERROR_FILE="$RESULT_DIR/error.log"

# without annotation
echo spliceai-toolkit predict -m "$MODEL_PATH" -o "$OUTPUT_PATH" -f $FLANKING_SIZE -i "$DATA_PATH" -t $THRESHOLD -D \
> "$OUTPUT_FILE" 2> "$ERROR_FILE"
spliceai-toolkit predict -m "$MODEL_PATH" -o "$OUTPUT_PATH" -f $FLANKING_SIZE -i "$DATA_PATH" -t $THRESHOLD -D \
> "$OUTPUT_FILE" 2> "$ERROR_FILE"


# with annotation
# echo spliceai-toolkit predict -m "$MODEL_PATH" -o "$OUTPUT_PATH" -f $FLANKING_SIZE -i "$DATA_PATH" -a "$ANNOTATION_PATH" -t $THRESHOLD -D \
# > "$OUTPUT_FILE" 2> "$ERROR_FILE"
# spliceai-toolkit predict -m "$MODEL_PATH" -o "$OUTPUT_PATH" -f $FLANKING_SIZE -i "$DATA_PATH" -a "$ANNOTATION_PATH" -t $THRESHOLD -D \
# > "$OUTPUT_FILE" 2> "$ERROR_FILE"

