#!/bin/bash

OUTPUT_DIR="../results/create-data/"
mkdir -p $OUTPUT_DIR

########################################
# Creating MANE dataset
########################################
openspliceai create-data \
--genome-fasta  ../data/GRCh38_chr21_chr22.fna \
--annotation-gff ../data/MANE.GRCh38.v1.3.refseq_genomic_chr21_chr22.gff \
--output-dir $OUTPUT_DIR \
--parse-type canonical
