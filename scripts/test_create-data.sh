#!/bin/bash

OUTPUT_DIR="../results/create-data/"
mkdir -p $OUTPUT_DIR
OUTPUT_FILE="$OUTPUT_DIR/output.log"
ERROR_FILE="$OUTPUT_DIR/error.log"

########################################
# Creating MANE dataset
########################################
openspliceai create-data \
--min-identity 0.8 \
--min-coverage 0.01 \
--genome-fasta  ../data/GRCh38_chr21_chr22.fna \
--annotation-gff ../data/MANE.GRCh38.v1.3.refseq_genomic_chr21_chr22.gff \
--output-dir $OUTPUT_DIR \
--parse-type canonical > $OUTPUT_FILE 2> $ERROR_FILE
