CL_max=10000
# Maximum nucleotide context length (CL_max/2 on either side of the 
# position of interest)
# CL_max should be an even number

SL=5000
# Sequence length of SpliceAIs (SL+CL will be the input length and
# SL will be the output length)

data_dir='../../results/train_test_dataset_spliceAI/'
splice_table='../../results/train_test_dataset_spliceAI/canonical_dataset.txt'
# ref_genome='/genomes/Homo_sapiens/UCSC/hg19/Sequence/WholeGenomeFasta/genome.fa'
ref_genome='/Users/chaokuan-hao/Documents/Projects/ref_genome/homo_sapiens/hg19/hg19.fa'
# Input details
sequence='../../results/train_test_dataset_spliceAI/canonical_sequence.txt'

# Output details
