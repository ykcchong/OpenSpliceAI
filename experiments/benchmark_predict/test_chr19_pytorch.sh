#!/bin/bash

cd /ccb/cybertron/smao10/openspliceai
 
NATIVE_DIR="./experiments/benchmark_predict"

# Run the setup script
python setup.py install

############################################################

# INDEPENDENT VARS
FLANKING_SIZE=80 # 80, 400, 2000, 10000
USE_ANNOTATION=1

############################################################

MODEL_PATH="./models/spliceai-mane/${FLANKING_SIZE}nt/model_${FLANKING_SIZE}nt_rs42.pt"
OUTPUT_PATH="$NATIVE_DIR/results/pytorch_chr19_${FLANKING_SIZE}nt_anno${USE_ANNOTATION}"
DATA_PATH="./data/toy/human/chr19.fa"

ANNOTATION_PATH="./data/toy/human/chr19.gff"

THRESHOLD=0.9

OUTPUT_FILE="$OUTPUT_PATH/output.log"
ERROR_FILE="$OUTPUT_PATH/error.log"

#############################################################

mkdir -p "$OUTPUT_PATH"
SCALENE_COMMAND="scalene --outfile $OUTPUT_PATH/scalene.html"

if [ "$USE_ANNOTATION" -eq 1 ]; then
    # with annotation
    COMMAND="$SCALENE_COMMAND --- ./experiments/benchmark_predict/predict_test.py -m $MODEL_PATH -o $OUTPUT_PATH -f $FLANKING_SIZE -i $DATA_PATH -a $ANNOTATION_PATH -t $THRESHOLD > $OUTPUT_FILE 2> $ERROR_FILE"
else
    # without annotation
    COMMAND="$SCALENE_COMMAND --- ./experiments/benchmark_predict/predict_test.py -m $MODEL_PATH -o $OUTPUT_PATH -f $FLANKING_SIZE -i $DATA_PATH -t $THRESHOLD > $OUTPUT_FILE 2> $ERROR_FILE"
fi

# Echo the command to verify it
echo $COMMAND

# Execute the command
eval $COMMAND