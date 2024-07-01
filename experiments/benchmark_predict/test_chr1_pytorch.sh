#!/bin/bash

# Benchmark task: predict on chr1 
# Params: flanking size (80, 400, 2000, 10000), without vs with annotation, pytorch vs. keras spliceai model

cd /ccb/cybertron/smao10/openspliceai
 
NATIVE_DIR="./experiments/benchmark_predict"

# Run the setup script
python setup.py install

############################################################

# INDEPENDENT VARS
FLANKING_SIZE=10000
USE_ANNOTATION=1

############################################################

MODEL_PATH="./models/spliceai-mane/${FLANKING_SIZE}nt/model_${FLANKING_SIZE}nt_rs42.pt"
OUTPUT_PATH="$NATIVE_DIR/results/pytorch"
DATA_PATH="./data/ref_genome/homo_sapiens/GRCh38/chr1_extracted.fna"

ANNOTATION_PATH="./data/toy/human/chr1.gff"

THRESHOLD=0.9

OUTPUT_FILE="$OUTPUT_PATH/output.log"
ERROR_FILE="$OUTPUT_PATH/error.log"

#############################################################

SCALENE_COMMAND="scalene --html --outfile $NATIVE_DIR/scalene.out"

if [ "$USE_ANNOTATION" -eq 1 ]; then
    # with annotation
    echo "$SCALENE_COMMAND python ./experiments/benchmark_predict/predict_test.py -m \"$MODEL_PATH\" -o \"$OUTPUT_PATH\" -f $FLANKING_SIZE -i \"$DATA_PATH\" -a \"$ANNOTATION_PATH\" -t $THRESHOLD -D > \"$OUTPUT_FILE\" 2> \"$ERROR_FILE\""

    $SCALENE_COMMAND python ./experiments/benchmark_predict/predict_test.py -m "$MODEL_PATH" -o "$OUTPUT_PATH" -f $FLANKING_SIZE -i "$DATA_PATH" -a "$ANNOTATION_PATH" -t $THRESHOLD -D \
    > "$OUTPUT_FILE" 2> "$ERROR_FILE"
else
    # without annotation
    echo "$SCALENE_COMMAND python ./spliceaitoolkit/predict/predict_test.py -m \"$MODEL_PATH\" -o \"$OUTPUT_PATH\" -f $FLANKING_SIZE -i \"$DATA_PATH\" -t $THRESHOLD -D > \"$OUTPUT_FILE\" 2> \"$ERROR_FILE\""

    $SCALENE_COMMAND python ./spliceaitoolkit/predict/predict_test.py -m "$MODEL_PATH" -o "$OUTPUT_PATH" -f $FLANKING_SIZE -i "$DATA_PATH" -t $THRESHOLD -D \
    > "$OUTPUT_FILE" 2> "$ERROR_FILE"
fi