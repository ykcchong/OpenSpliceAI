#CL_set = [80, 400, 2000, 10000]
#SL_set = [2500, 5000, 7500]

CL_max=80
# Maximum nucleotide context length (CL_max/2 on either side of the 
# position of interest)
# CL_max should be an even number

SL=5000
# Sequence length of SpliceAIs (SL+CL will be the input length and
# SL will be the output length)

# step 1 input
# ref_genome='/genomes/Homo_sapiens/UCSC/hg19/Sequence/WholeGenomeFasta/genome.fa'
# ref_genome='/Users/chaokuan-hao/Documents/Projects/ref_genome/homo_sapiens/hg19/hg19.fa'
ref_genome='/Users/alanmao/Desktop/Research/spliceAI-toolkit/train_data/human/hg19.fa'


# step 1 output dir
# data_dir='../../results/train_test_dataset_spliceAI/'
# splice_table='../../results/train_test_dataset_spliceAI/canonical_dataset.txt'
data_dir='../../results/spliceai_test/'
splice_table='../../results/spliceai_test/canonical_dataset.txt'

# Input details
# sequence='../../results/train_test_dataset_spliceAI/canonical_sequence.txt'
sequence='../../results/spliceai_test/canonical_sequence.txt'

# Output details
