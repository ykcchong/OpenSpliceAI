#!/bin/bash

source constants.py

# computing the left and right window size
# NOTE: first nucleotide not included by BEDtools
CLr=$((CL_max/2))
CLl=$(($CLr+1))

# stores relevant information from the splice table into temp bed file
cat $splice_table | awk -v CLl=$CLl -v CLr=$CLr '{print $3"\t"($5-CLl)"\t"($6+CLr)}' > temp.bed

# error checking
awk '$2 >= 0 && $3 > $2' temp.bed > filtered.bed

# convert into a sequence file 
bedtools getfasta -bed filtered.bed -fi $ref_genome -fo $sequence -tab

# clean temp files
rm temp.bed filtered.bed
