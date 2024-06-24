#!/bin/bash

# short example with chr19 -> tests naming with short header lines, splitting fasta entry, prediction by gene extraction without annotation file, hdf file usage

SETUP="/ccb/cybertron/smao10/openspliceai/setup.py"
RESULT_DIR="./results/predict/SpliceAI_5000_400"

# Run the setup script
python "$SETUP" install
if [ -d "$RESULT_DIR" ]; then
    rm -rf "$RESULT_DIR"
fi
mkdir -p $RESULT_DIR

MODEL_PATH="./models/spliceai-mane/400nt/model_400nt_rs40.pt"
OUTPUT_PATH="./results/predict"
DATA_PATH="./data/toy/human/chr19.fa"
FLANKING_SIZE=400
THRESHOLD=0.9

OUTPUT_FILE="$RESULT_DIR/output.log"
ERROR_FILE="$RESULT_DIR/error.log"

# Run the spliceai-toolkit predict command
echo spliceai-toolkit predict -m "$MODEL_PATH" -o "$OUTPUT_PATH" -f $FLANKING_SIZE -i "$DATA_PATH" -t $THRESHOLD -D \
> "$OUTPUT_FILE" 2> "$ERROR_FILE"
spliceai-toolkit predict -m "$MODEL_PATH" -o "$OUTPUT_PATH" -f $FLANKING_SIZE -i "$DATA_PATH" -t $THRESHOLD -D \
> "$OUTPUT_FILE" 2> "$ERROR_FILE"