'''Script to perform many kinds of extractions given set of donor and acceptor sequences'''

import random
from pyfaidx import Fasta
import gffutils
import os

def sample_fasta_entries(input_file, output_file, num_entries=10, random_seed=42):
    '''Randomly sample a specified number of entries from a FASTA file'''
    
    # Read the input fasta file
    fasta = Fasta(input_file, duplicate_action='first')

    # Get the list of sequence names
    sequence_names = list(fasta.keys())

    # Set the random seed
    random.seed(random_seed)

    # Randomly select the specified number of entries
    selected_entries = random.sample(sequence_names, num_entries)

    # Write the selected entries to the output file
    with open(output_file, 'w') as f:
        for entry in selected_entries:
            f.write('>' + fasta[entry].name + '\n')
            f.write(fasta[entry][:].seq + '\n')
            
def split_fasta(input_fasta, output_base, num_files, first_n=None):
    '''Split a FASTA file into multiple files'''
    
    fasta = Fasta(input_fasta, duplicate_action='first')
    if first_n:
        sequence_names = list(fasta.keys())[:first_n]
    else: 
        sequence_names = list(fasta.keys())
    num_entries = len(sequence_names)
    entries_per_file = num_entries // num_files
    print(f"Splitting {num_entries} entries into {num_files} files with {entries_per_file} entries each.")
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    for i in range(num_files):
        start = i * entries_per_file
        end = (i + 1) * entries_per_file
        if i == num_files - 1:
            end = num_entries
        output_file = f'{output_base}_batch{i+1}.fa'
        with open(output_file, 'w') as f:
            for entry in sequence_names[start:end]:
                f.write('>' + fasta[entry].name + '\n')
                f.write(fasta[entry][:].seq + '\n')

def specific_extract(chrom, start, stop):
    '''Extract a specific sequence from the reference genome'''
    
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


### SAMPLE 7 - 300-entry random sample of 400-bp donor and acceptor segments
# num_entries = 300
# sample_num = 7
# dataset = 'test'
# base = '/ccb/cybertron/smao10/openspliceai/experiments/mutagenesis/experiment_2/data'
# sample_fasta_entries(f'{base}/{dataset}_acceptor.fa', f'{base}/acceptor_{sample_num}.fa', num_entries)
# sample_fasta_entries(f'{base}/{dataset}_donor.fa', f'{base}/donor_{sample_num}.fa', num_entries)

### Splitting Sample 7 into 10 sets of 10 transcripts for first 100 segments
first_n = 100
input_sample = 7
base = '/ccb/cybertron/smao10/openspliceai/experiments/mutagenesis/experiment_2/data'
split_fasta(f'{base}/acceptor_{input_sample}.fa', f'{base}/keras_job/acceptor', 10, first_n)
split_fasta(f'{base}/donor_{input_sample}.fa', f'{base}/keras_job/donor', 10, first_n)

# specific_extract(3, 142740137, 142740263)