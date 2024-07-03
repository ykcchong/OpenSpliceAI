'''
Default SpliceAI prediction for benchmarking using Keras model.
- derived from SpliceAI repository
- uses all 5 models and averages scores
'''
import argparse
import numpy as np
from keras.models import load_model
from pyfaidx import Fasta
import h5py
import os
import re
import sys

####################
### FROM PYTORCH ###
####################

def initialize_globals(flanking_size, split_fasta_threshold=1500000):
    
    assert int(flanking_size) in [80, 400, 2000, 10000]
    
    global CL_max                  # context length for sequence prediction (flanking size sum)
    global SPLIT_FASTA_THRESHOLD   # maximum length of fasta entry before splitting
    global HDF_THRESHOLD_LEN       # maximum size before reading sequence into an HDF file for storage
    
    CL_max = flanking_size
    SPLIT_FASTA_THRESHOLD = split_fasta_threshold
    HDF_THRESHOLD_LEN = 0

def process_gff(fasta_file, gff_file, output_dir):
    """
    Processes a GFF file to extract gene regions and creates a new FASTA file
    with the extracted gene sequences.

    Parameters:
    - gff_file: Path to the GFF file.
    - fasta_file: Path to the input FASTA file.
    - output_fasta_file: Path to the output FASTA file.
    """
    # define output FASTA file
    output_fasta_file = f'{output_dir}{os.path.splitext(os.path.basename(fasta_file))[0]}_genes.fa'

    # read the input FASTA file
    fasta = Fasta(fasta_file)

    # open the output FASTA file for writing
    count = 0
    with open(output_fasta_file, 'w') as output_fasta:
        # read the GFF file
        with open(gff_file, 'r') as gff:
            for line in gff:
                if line.startswith('#'):
                    continue
                
                # split the GFF line into fields
                fields = line.strip().split('\t')
                if len(fields) < 9:
                    print(f'\t[ERR] line does not have enough fields:\n{line}')
                    continue  # lines with not enough fields are erroneous? 
                
                # extract relevant information from the GFF fields
                seqid = fields[0]
                feature_type = fields[2]
                start = int(fields[3])
                end = int(fields[4])
                strand = fields[6]
                attributes = fields[8]

                # process only gene features
                if feature_type != 'gene':
                    continue

                # extract gene ID, or if not then Name, from the attributes
                gene_id = 'unknown_gene'
                for attribute in attributes.split(';'):
                    if attribute.startswith('ID=') or attribute.startswith('Name='):
                        gene_id = attribute.split('=')[1]
                        break

                # extract the gene sequence from the FASTA file
                sequence = fasta[seqid][start-1:end]  # adjust for 0-based indexing

                # reverse complement the sequence if on the negative strand
                if strand == '-':
                    sequence = sequence.reverse.complement

                # write the gene sequence to the output FASTA file
                output_fasta.write(f'>{gene_id} {seqid}:{start}-{end}({strand})\n')
                output_fasta.write(str(sequence) + '\n')
                count += 1

    print(f"\t[INFO] {count} gene sequences have been extracted to {output_fasta_file}")

    return output_fasta_file

def split_fasta(genes, split_fasta_file):
    '''
    Splits any long genes in the given Fasta object into segments of SPLIT_FASTA_THRESHOLD length and writes them to a FASTA file.

    Parameters:
    - genes (Fasta): A pyfaidx dictionary-like object containing gene records.
    - split_fasta_file (str): The path to the output FASTA file.
    '''
    name_pattern = re.compile(r'(.*)(chr[a-zA-Z0-9_]*):(\d+)-(\d+)\(([-+])\)')
    chrom_pattern = re.compile(r'(chr[a-zA-Z0-9_]+)')
    
    def create_name(record, start_pos, end_pos):
        '''
        Write a line of the split FASTA file.
        
        Params:
        - record: Gene record
        - start_pos: Relative start position of the segment (1-indexed)
        - end_pos: Relative end position of the segment (1-indexed)
        '''
        
        # default extended name
        segment_name = record.long_name
        
        # search for the pattern in the name
        match = name_pattern.search(segment_name)
        if match:
            prefix = match.group(1)
            chromosome = match.group(2)
            strand = match.group(5)
            abs_start = int(match.group(3))
            
            # compute true absolute start and end positions
            start = abs_start - 1 + start_pos
            end = abs_start - 1 + end_pos
                        
            segment_name = f"{prefix}{chromosome}:{start}-{end}({strand})"
            
        else:
            chrom_match = chrom_pattern.search(segment_name)
            if chrom_match:
                seqid = chrom_match.group(1) # use chromosome to denote sequence ID
            else:
                seqid = record.name # use original name to denote sequence ID (NOTE: must be unique for each sequence in the FASTA file)        
            
            strand = '.' # NOTE: unknown strands will be treated as a forward strand further downstream (supply neg_strands argument to get_sequences() to override)
            
            # construct the fixed string with the split denoted
            segment_name = f"{seqid}:{start_pos}-{end_pos}({strand})"
        
        return segment_name
                    
    with open(split_fasta_file, 'w') as output_file:
        for record in genes:
            seq_length = len(genes[record.name])
            if seq_length > SPLIT_FASTA_THRESHOLD:
                # process each segment into a new entry
                for i in range(0, seq_length, SPLIT_FASTA_THRESHOLD):
                    
                    # obtain the split sequence (with flanking to preserve predictions across splits)
                    start_slice = i - (CL_max // 2) if i - (CL_max // 2) >= 0 else 0
                    end_slice = i + SPLIT_FASTA_THRESHOLD + (CL_max // 2) if i + SPLIT_FASTA_THRESHOLD + (CL_max // 2) <= seq_length else seq_length
                    segment_seq = genes[record.name][start_slice:end_slice].seq # added flanking sequences to preserve predictions 
                    
                    # formulate the sequence name using pattern matching
                    segment_name = create_name(record, start_slice+1, end_slice)
                    
                    output_file.write(f">{segment_name}\n")
                    output_file.write(f"{segment_seq}\n")
                    
            else:
                # write sequence as is (still ensuring name in format)
                segment_seq = genes[record.name][:]
                segment_name = create_name(record, 1, len(segment_seq))
            
                output_file.write(f">{segment_name}\n")
                output_file.write(f"{segment_seq}\n")
    

def get_sequences(fasta_file, output_dir, neg_strands=None, debug=False):
    """
    Extract sequences for each protein-coding gene, process them based on strand orientation,
    and save data in file depending on the sequence size (HDF file if sequence >HDF_THRESHOLD_LEN, 
    else temp file).

    Parameters:
    - fasta_file: Path to the FASTA file.
    - output_dir: Directory to save the output files.
    - neg_strands (list): List of IDs for fasta entries where the sequence is on the reverse strand. Default is None.

    Returns:
    - Path to datafile.
    """

    # detect sequence length, determine file saving method to use
    total_length = 0
    use_hdf = False
    need_splitting = False

    # NOTE: always creates uppercase sequence, uses [1,0]-indexed sequence names (does not affect slicing), takes simple name from FASTA
    genes = Fasta(fasta_file, one_based_attributes=True, read_long_names=False, sequence_always_upper=True) 

    for record in genes:
        record_length = len(genes[record.name])
        total_length += record_length
        if not use_hdf and total_length > HDF_THRESHOLD_LEN:
            use_hdf = True
            print(f'\t[INFO] Input FASTA sequences over {HDF_THRESHOLD_LEN}: use_hdf = True.')
        if not need_splitting and record_length > SPLIT_FASTA_THRESHOLD:
            need_splitting = True
            print(f'\t[INFO] Input FASTA contains sequence(s) over {SPLIT_FASTA_THRESHOLD}: need_splitting = True')
        if use_hdf and need_splitting:
            break
    
    if need_splitting:
        split_fasta_file = f'{output_dir}{os.path.splitext(os.path.basename(fasta_file))[0]}_split.fa'
        print(f'\t[INFO] Splitting {fasta_file}.')

        split_fasta(genes, split_fasta_file)

        # re-loads the pyfaidx Fasta object with split genes
        genes = Fasta(split_fasta_file, one_based_attributes=True, read_long_names=True, sequence_always_upper=True) # need long name to handle duplicate seqids after splits
        print(f"\t[INFO] Saved and loaded {split_fasta_file}.")

    NAME = [] # Gene Header
    SEQ  = [] # Sequences

    # obtain the headers and sequences from FASTA file
    for record in genes:
        seq_id = record.long_name
        sequence = genes[record.name][:].seq    
        
        # reverse strand if explicitly specified, name with strand info
        if neg_strands is not None and record.name in neg_strands: 
            seq_id = str(seq_id) + ':-'
            sequence = sequence.reverse.complement
        else:
            seq_id = str(seq_id) + ':+'
        
        NAME.append(seq_id)
        SEQ.append(str(sequence))
    
    # write the sequences to datafile
    if use_hdf:
        datafile_path = f'{output_dir}datafile.h5'
        dt = h5py.string_dtype(encoding='utf-8')
        with h5py.File(datafile_path, 'w') as datafile: # hdf5 information file
            datafile.create_dataset('NAME', data=np.asarray(NAME, dtype=dt), dtype=dt)
            datafile.create_dataset('SEQ', data=np.asarray(SEQ, dtype=dt), dtype=dt)
    else:
        datafile_path = f'{output_dir}datafile.txt'
        with open(datafile_path, 'w') as datafile: # temp sequence file
            for name, seq in zip(NAME, SEQ):
                datafile.write(f'{name}\n{seq}\n')
    
    if debug:
        print(f'\t[DEBUG] len(NAME): {len(NAME)}, len(SEQ): {len(SEQ)}', file=sys.stderr)
    
    return datafile_path, NAME, SEQ



#####################
### FROM SPLICEAI ###
#####################

# One-hot encoding function
def one_hot_encode(seq):
    map = np.asarray([[0, 0, 0, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

    seq = seq.upper().replace('A', '\x01').replace('C', '\x02')
    seq = seq.replace('G', '\x03').replace('T', '\x04').replace('N', '\x00')

    return map[np.fromstring(seq, np.int8) % 5]

# Normalizing chromosome name
def normalise_chrom(source, target):
    def has_prefix(x):
        return x.startswith('chr')

    if has_prefix(source) and not has_prefix(target):
        return source.strip('chr')
    elif not has_prefix(source) and has_prefix(target):
        return 'chr'+source

    return source

# Prediction function - derived from SpliceAI get_delta_score()




def get_models(flanking_size):
    return [load_model(f'./models/SpliceAI/SpliceNet{flanking_size}_c{i}.h5') for i in range(1, 6)]

# Example usage
if __name__ == "__main__":
    
    parser_predict = argparse.ArgumentParser(description='Predicts splice sites using SpliceAI model')
    
    # parser_predict.add_argument('--model', '-m', type=str, default="SpliceAI", help='Path to a PyTorch SpliceAI model file or "SpliceAI" for the default model')
    parser_predict.add_argument('--output-dir', '-o', type=str, required=True, help='Output directory to save the data')
    parser_predict.add_argument('--flanking-size', '-f', type=int, default=80, help='Sum of flanking sequence lengths on each side of input (i.e. 40+40)')
    parser_predict.add_argument('--input-sequence', '-i', type=str, help="Path to FASTA file of the input sequence")
    parser_predict.add_argument('--annotation-file', '-a', type=str, required=False, help="Path to GFF file of coordinates for genes")
    parser_predict.add_argument('--threshold', '-t', type=float, default=1e-6, help="Threshold to determine acceptor and donor sites")
    # parser_predict.add_argument('--predict-all', '-p', action='store_true', required=False, help="Writes all collected predictions to an intermediate file (Warning: on full genomes, will consume much space.)")
    # parser_predict.add_argument('--debug', '-D', action='store_true', required=False, help="Run in debug mode (debug statements are printed to stderr)")
    # parser_predict.add_argument('--hdf-threshold', type=int, default=0, help='Maximum size before reading sequence into an HDF file for storage')
    # parser_predict.add_argument('--flush-threshold', type=int, default=500, help='Maximum number of predictions before flushing to file')
    parser_predict.add_argument('--split-threshold', type=int, default=1500000, help='Maximum length of FASTA entry before splitting')
    # parser_predict.add_argument('--chunk-size', type=int, default=100, help='Chunk size for loading HDF5 dataset')
    
    args = parser_predict.parse_args()
    
    initialize_globals(args.flanking_size, args.split_threshold)

    models = get_models(args.flanking_size)
    fasta_file = args.input_sequence
    annotation_file = args.annotation_file
    threshold = args.threshold
    output_dir = args.output_dir

    # Make predictions
    predictions