#!/bin/bash

# short example of grch38 with annotation file of first 100000 lines of gff file -> tests prediction with annotation file, naming with long header, hdf file usage

SETUP="/ccb/cybertron/smao10/openspliceai/setup.py"
RESULT_DIR="./results/predict/test2/SpliceAI_5000_400"
mkdir -p $RESULT_DIR

# Run the setup script
pip install .

MODEL_PATH="./models/spliceai-mane/400nt/model_400nt_rs14.pt"
OUTPUT_PATH="./results/predict/test2"
DATA_PATH="./data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna"
ANNOTATION_PATH="./data/toy/human/test.gff"
FLANKING_SIZE=400
THRESHOLD=0.9

OUTPUT_FILE="$RESULT_DIR/output.log"
ERROR_FILE="$RESULT_DIR/error.log"

# Run the openspliceai predict command
echo openspliceai predict -m "$MODEL_PATH" -o "$OUTPUT_PATH" -f $FLANKING_SIZE -i "$DATA_PATH" -a "$ANNOTATION_PATH" -t $THRESHOLD -D \
> "$OUTPUT_FILE" 2> "$ERROR_FILE"
openspliceai predict -m "$MODEL_PATH" -o "$OUTPUT_PATH" -f $FLANKING_SIZE -i "$DATA_PATH" -a "$ANNOTATION_PATH" -t $THRESHOLD -D \
> "$OUTPUT_FILE" 2> "$ERROR_FILE"