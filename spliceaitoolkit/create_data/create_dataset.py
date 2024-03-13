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
import os
import time
from spliceaitoolkit.constants import *
from spliceaitoolkit.create_data.utils import ceil_div, one_hot_encode
import argparse

def replace_non_acgt_to_n(input_string):
    """
    Use a generator expression to go through each character in the input string.
    If the character is in the set of allowed characters, keep it as is.
    Otherwise, replace it with 'N'.

    Parameters:
    - input_string (str): The nucleotide sequence.

    Returns:
    - str: The modified sequence with non-ACGT nucleotides replaced by 'N'.
    """

    # Define the set of allowed characters
    allowed_chars = {'A', 'C', 'G', 'T'}    
    return ''.join(char if char in allowed_chars else 'N' for char in input_string)


def create_datapoints(seq, strand, label):
    """
    This function first converts the sequence into an integer array, where
    A, C, G, T, N are mapped to 1, 2, 3, 4, 0 respectively. If the strand is
    negative, then reverse complementing is done. The labels 
    are directly used as they are, converted into an array of integers,
    where 0, 1, 2 correspond to no splicing, acceptor, donor 
    respectively. It then calls reformat_data and one_hot_encode
    and returns X, Y which can be used by Pytorch Model.

    Parameters:
    - seq (str): The nucleotide sequence.
    - strand (str): The strand information ('+' or '-').
    - label (str): A string representation of labels for each nucleotide.

    Returns:
    - tuple: A tuple containing the one-hot encoded sequence and labels.
    """

    # No need to reverse complement the sequence, as sequence is already reverse complemented from previous step
    seq = 'N' * (CL_max // 2) + seq + 'N' * (CL_max // 2)
    seq = seq.upper().replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')

    # Convert label string to array of integers
    label_array = np.array(list(map(int, list(label))))
    X0 = np.asarray(list(map(int, list(seq))))
    Y0 = label_array
    Y0 = [Y0]
    Xd, Yd = reformat_data(X0, Y0)
    X, Y = one_hot_encode(Xd, Yd)

    return X, Y

def reformat_data(X0, Y0):
    """
    Reformat sequence and label data into fixed-size blocks for processing.
    This function converts X0, Y0 of the create_datapoints function into
    blocks such that the data is broken down into data points where the
    input is a sequence of length SL+CL_max corresponding to SL nucleotides
    of interest and CL_max context nucleotides, the output is a sequence of
    length SL corresponding to the splicing information of the nucleotides
    of interest. The CL_max context nucleotides are such that they are
    CL_max/2 on either side of the SL nucleotides of interest.

    Parameters:
    - X0 (numpy.ndarray): Original sequence data as an array of integer encodings.
    - Y0 (list of numpy.ndarray): Original label data as a list containing a single array of integer encodings.

    Returns:
    - numpy.ndarray: Reformatted sequence data.
    - list of numpy.ndarray: Reformatted label data, wrapped in a list.
    """
    # Calculate the number of data points needed
    num_points = ceil_div(len(Y0[0]), SL)
    # Initialize arrays to hold the reformatted data
    Xd = np.zeros((num_points, SL + CL_max))
    Yd = [-np.ones((num_points, SL)) for _ in range(1)]
    # Pad the sequence and labels to ensure divisibility
    X0 = np.pad(X0, (0, SL), 'constant', constant_values=0)
    Y0 = [np.pad(Y0[t], (0, SL), 'constant', constant_values=-1) for t in range(1)]

    # Fill the initialized arrays with data in blocks
    for i in range(num_points):
        Xd[i] = X0[SL * i : SL * (i + 1) + CL_max]
        Yd[0][i] = Y0[0][SL * i : SL * (i + 1)]

    return Xd, Yd                  

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
            SEQ = h5f['SEQ'][:]
            LABEL = h5f['LABEL'][:]

        print(f"\tWriting {output_file} ... ")
        with h5py.File(output_file, 'w') as h5f2:
            seq_num = len(SEQ)
            CHUNK_SIZE = 100

            print("seq_num: ", seq_num)
            print("STRAND.shape[0]: ", STRAND.shape[0])
            print("TX_START.shape[0]: ", TX_START.shape[0])
            print("TX_END.shape[0]: ", TX_END.shape[0])
            print("LABEL.shape[0]: ", LABEL.shape[0])

            # Create dataset
            for i in range(seq_num // CHUNK_SIZE):

                # Each dataset has CHUNK_SIZE genes
                if (i+1) == seq_num // CHUNK_SIZE:
                    NEW_CHUNK_SIZE = CHUNK_SIZE + seq_num%CHUNK_SIZE
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
                    X, Y = create_datapoints(fixed_seq, strand_decode, label_decode)   

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