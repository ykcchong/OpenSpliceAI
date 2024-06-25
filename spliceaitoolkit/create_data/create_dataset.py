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
import tqdm
import time
from spliceaitoolkit.constants import *
from spliceaitoolkit.create_data.utils import ceil_div, replace_non_acgt_to_n, create_datapoints
import argparse          

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
    for data_type in ['train', 'test']:
        print(("\tProcessing %s ..." % data_type))
        input_file = f"{args.output_dir}/datafile_{data_type}.h5"
        output_file = f"{args.output_dir}/dataset_{data_type}.h5"

        print(f"\tReading {input_file} ... ")
        with h5py.File(input_file, 'r') as h5f:
            SEQ = h5f['SEQ'][:]
            LABEL = h5f['LABEL'][:]
            STRAND = h5f['STRAND'][:]
            TX_START = h5f['TX_START'][:]
            TX_END = h5f['TX_END'][:]
            # SEQ = h5f['SEQ'][:]
            # LABEL = h5f['LABEL'][:]

        print(f"\tWriting {output_file} ... ")
        with h5py.File(output_file, 'w') as h5f2:
            seq_num = len(SEQ)
            CHUNK_SIZE = 100

            print("seq_num: ", seq_num)
            print("STRAND.shape[0]: ", STRAND.shape[0])
            print("TX_START.shape[0]: ", TX_START.shape[0])
            print("TX_END.shape[0]: ", TX_END.shape[0])
            print("LABEL.shape[0]: ", LABEL.shape[0])

            # create dataset
            num_chunks = ceil_div(seq_num, CHUNK_SIZE) # ensures that even if seq_num < CHUNK_SIZE, will still create a chunk
            for i in tqdm(range(num_chunks), desc='Processing chunks...'):

                # each dataset has CHUNK_SIZE genes
                if i == num_chunks - 1: # if last chunk, process remainder or full chunk size if no remainder
                    NEW_CHUNK_SIZE = seq_num % CHUNK_SIZE or CHUNK_SIZE 
                else:
                    NEW_CHUNK_SIZE = CHUNK_SIZE

                X_batch, Y_batch = [], [[] for _ in range(1)]

                for j in range(NEW_CHUNK_SIZE):
                    idx = i*CHUNK_SIZE + j

                    seq_decode = SEQ[idx].decode('ascii')
                    strand_decode = STRAND[idx].decode('ascii')
                    tx_start_decode = TX_START[idx].decode('ascii')
                    tx_end_decode = TX_END[idx].decode('ascii')
                    label_decode = LABEL[idx].decode('ascii')

                    fixed_seq = replace_non_acgt_to_n(seq_decode)
                    X, Y = create_datapoints(fixed_seq, label_decode, CL_max=args.flanking_size)
                    print('shapes', X.shape, Y.shape)   

                    X_batch.extend(X)
                    Y_batch[0].extend(Y[0])

                # Convert batches to arrays and save as HDF5
                X_batch = np.asarray(X_batch).astype('int8')
                print("X_batch.shape: ", X_batch.shape)
                Y_batch[0] = np.asarray(Y_batch[0]).astype('int8')
                print("len(Y_batch[0]): ", len(Y_batch[0]))
                h5f2.create_dataset('X' + str(i), data=X_batch)
                h5f2.create_dataset('Y' + str(i), data=Y_batch)

    print("--- %s seconds ---" % (time.time() - start_time))