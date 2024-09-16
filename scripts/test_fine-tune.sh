#!/bin/bash

OUTPUT_DIR="../results/fine-tune/"
mkdir -p $OUTPUT_DIR
OUTPUT_FILE="$OUTPUT_DIR/output.log"
ERROR_FILE="$OUTPUT_DIR/error.log"

openspliceai fine-tune --flanking-size 80 \
--train-dataset ../results/create-data/dataset_train.h5 \
--test-dataset ../results/create-data/dataset_test.h5 \
--output-dir $OUTPUT_DIR \
--project-name human_MANE_chr21_chr22_test \
--pretrained-model  /home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/models/spliceai-mane/80nt/model_80nt_rs12.pt \
--random-seed 11 \
--loss cross_entropy_loss > $OUTPUT_FILE 2> $ERROR_FILE
