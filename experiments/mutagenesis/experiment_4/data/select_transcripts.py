'''Script to perform many kinds of extractions given set of donor and acceptor sequences'''

import random
from pyfaidx import Fasta
import gffutils
import os

def specific_extract_GRCh37(chrom, start, stop):
    base = '/ccb/cybertron/smao10/openspliceai'
    fasta_path = f'{base}/data/ref_genome/homo_sapiens/GRCh37/hg19.fa'
        
    # Load the FASTA file
    fasta = Fasta(fasta_path, sequence_always_upper=True, rebuild=False)
    
    # Get the sequence
    seq = str(fasta[f'chr{chrom}'][start-1:stop].seq)
    print(seq, len(seq))
    
    # Write the sequence to the output file
    with open(f'{chrom}_{start}_{stop}.fa', "w") as f:
        f.write(f">{chrom}_{start}_{stop}\n{seq}")

# include flanking
# specific_extract_GRCh37(3, 142740137-5000, 142740263+5000)

# TODO: finish
def specific_extract_GRCh38(chrom, start, stop, output_dir, strand='+'):
    base = '/ccb/cybertron/smao10/openspliceai'
    fasta_path = f'{base}/data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna'
        
    # Load the FASTA file
    fasta = Fasta(fasta_path, sequence_always_upper=True, rebuild=False)
    
    # Get the sequence
    record = fasta[f'chr{chrom}'][start-1:stop]
    if strand == '-':
        print('reverse')
        seq = str(record.reverse.complement.seq)
    else:
        seq = str(record.seq)
    print(seq, len(seq))
    
    # Write the sequence to the output file
    with open(f'{base}/{output_dir}/{chrom}_{start}_{stop}.fa', "w") as f:
        f.write(f">{chrom}_{start}_{stop}\n{seq}")
        
# E2F1 -> chr20: exon 4, 5 ATAC
# donor: 33,678,201
# acceptor: 33,677,540 

#specific_extract_GRCh38(20, 33678201-20-5000, 33678201+20+5000, 'experiments/mutagenesis/experiment_4/data', '-')
specific_extract_GRCh38(20, 33677540-20-5000, 33677540+20+5000, 'experiments/mutagenesis/experiment_4/data', '-')

# def full_extract(chrom, start, stop):
#     '''Get the full 15k input sequence for prediction'''
    
#     base = '/ccb/cybertron/smao10/openspliceai'
#     fasta_path = f'{base}/data/ref_genome/homo_sapiens/GRCh37/hg19.fa'
        
#     # Load the FASTA file
#     fasta = Fasta(fasta_path, sequence_always_upper=True, rebuild=False)
    
#     # Get the sequence
#     seq = str(fasta[f'chr{chrom}'][start-1:stop].seq)
#     print(seq, len(seq))
    
#     # Write the sequence to the output file
#     with open(f'{chrom}_{start}_{stop}.fa', "w") as f:
#         f.write(f">{chrom}_{start}_{stop}\n{seq}")