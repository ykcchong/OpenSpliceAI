#!/bin/bash

OUTPUT_DIR="../results/train/"
mkdir -p $OUTPUT_DIR
OUTPUT_FILE="$OUTPUT_DIR/output.log"
ERROR_FILE="$OUTPUT_DIR/error.log"

openspliceai train --flanking-size 80 \
--train-dataset ../results/create-data/dataset_train.h5 \
--test-dataset ../results/create-data/dataset_test.h5 \
--output-dir $OUTPUT_DIR \
--project-name human_MANE_chr21_chr22_test \
--random-seed 11 \
--loss focal_loss > $OUTPUT_FILE 2> $ERROR_FILE
