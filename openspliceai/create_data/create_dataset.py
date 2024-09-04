"""
create_dataset.py

- Converts sequences to one-hot encoded format, considering strand information
- Pads sequences and replaces non-standard nucleotides
- Breaks down data into manageable chunks for processing
- Counts occurrences of motifs at donor and acceptor splice sites

Usage:
    Required arguments:
        - output_dir: Directory where the output files will be saved.

Example:
    python create_dataset.py --output_dir path/to/output

Functions:
    replace_non_acgt_to_n(input_string): Replace non-ACGT characters in sequences with 'N'.
    create_datapoints(seq, strand, label): Prepare data points for model training or evaluation.
    reformat_data(X0, Y0): Reformat raw data points into structured input and output arrays.
    create_dataset(args): Main function to process the input data and create datasets.
"""

import h5py
import numpy as np
from tqdm import tqdm
import time
from openspliceai.constants import *
from openspliceai.create_data.utils import ceil_div, replace_non_acgt_to_n, create_datapoints
import argparse          

CHUNK_SIZE = 100 # size of chunks to process data in

def create_dataset(args):
    """
    Create HDF5 datasets for training and testing from processed genomic data.

    This function reads processed genomic data from HDF5 files, encodes the data,
    and writes the encoded data into new HDF5 files formatted for use in machine learning models.
    The data is chunked to manage memory efficiently and enable batch processing during model training.

    Parameters:
    - args (argparse.Namespace): Command-line arguments
        - output_dir (str): The directory where the HDF5 files will be saved.
    """
    
    print("--- Step 2: Creating dataset.h5 ... ---")
    start_time = time.time()
    
    dataset_ls = [] 
    if args.chr_split == 'test':
        dataset_ls.append('test')
    elif args.chr_split == 'train-test':
        dataset_ls.append('test')
        dataset_ls.append('train')
    for dataset_type in dataset_ls:
        print(("\tProcessing %s ..." % dataset_type))
        if args.biotype =="non-coding":
            input_file = f"{args.output_dir}/datafile_{dataset_type}_ncRNA.h5"
            output_file = f"{args.output_dir}/dataset_{dataset_type}_ncRNA.h5"
        elif args.biotype =="protein-coding":
            input_file = f"{args.output_dir}/datafile_{dataset_type}.h5"
            output_file = f"{args.output_dir}/dataset_{dataset_type}.h5"

        print(f"\tReading {input_file} ... ")
        with h5py.File(input_file, 'r') as h5f:
            SEQ = h5f['SEQ'][:]
            LABEL = h5f['LABEL'][:]
            STRAND = h5f['STRAND'][:]
            TX_START = h5f['TX_START'][:]
            TX_END = h5f['TX_END'][:]

        print(f"\tWriting {output_file} ... ")
        with h5py.File(output_file, 'w') as h5f2:
            seq_num = len(SEQ)
            # create dataset
            num_chunks = ceil_div(seq_num, CHUNK_SIZE) # ensures that even if seq_num < CHUNK_SIZE, will still create a chunk
            for i in tqdm.tqdm(range(num_chunks), desc='Processing chunks...'):
                # each dataset has CHUNK_SIZE genes
                if i == num_chunks - 1: # if last chunk, process remainder or full chunk size if no remainder
                    NEW_CHUNK_SIZE = seq_num % CHUNK_SIZE or CHUNK_SIZE 
                else:
                    NEW_CHUNK_SIZE = CHUNK_SIZE
                X_batch, Y_batch = [], [[] for _ in range(1)]
                for j in range(NEW_CHUNK_SIZE):
                    idx = i*CHUNK_SIZE + j
                    seq_decode = SEQ[idx].decode('ascii')
                    label_decode = LABEL[idx].decode('ascii')
                    fixed_seq = replace_non_acgt_to_n(seq_decode)
                    X, Y = create_datapoints(fixed_seq, label_decode, CL_max=args.flanking_size)
                    X_batch.extend(X)
                    Y_batch[0].extend(Y[0])
                # Convert batches to arrays and save as HDF5
                X_batch = np.asarray(X_batch).astype('int8')
                Y_batch[0] = np.asarray(Y_batch[0]).astype('int8')

                h5f2.create_dataset('X' + str(i), data=X_batch)
                h5f2.create_dataset('Y' + str(i), data=Y_batch)

    print("--- %s seconds ---" % (time.time() - start_time))