###############################################################################
'''This code has functions to process sequences into .h5 datasets.'''
###############################################################################

import numpy as np
import torch
from math import ceil
from sklearn.metrics import average_precision_score
from spliceaitoolkit.constants import *

# One-hot encoding of the inputs: 
# 1: A;  2: C;  3: G;  4: T;  0: padding
IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

# One-hot encoding of the outputs: 
# 0: no splice;  1: acceptor;  2: donor;  -1: padding
OUT_MAP = np.asarray([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])

# Counting donor and acceptor motifs
donor_motif_counts = {}  
acceptor_motif_counts = {}  

def ceil_div(x, y):
    """
    Calculate the ceiling of a division between two numbers.

    Parameters:
    - x (int): Numerator
    - y (int): Denominator

    Returns:
    - int: The ceiling of the division result.
    """
    return int(ceil(float(x)/y))

def one_hot_encode(Xd, Yd):
    """
    Perform one-hot encoding on both the input sequence data (Xd) and the output label data (Yd) using
    predefined mappings (IN_MAP for inputs and OUT_MAP for outputs).

    Parameters:
    - Xd (numpy.ndarray): An array of integers representing the input sequence data where each nucleotide
        is encoded as an integer (1 for 'A', 2 for 'C', 3 for 'G', 4 for 'T', and 0 for padding).
    - Yd (list of numpy.ndarray): A list containing a single array of integers representing the output label data,
        where each label is encoded as an integer (0 for 'no splice', 1 for 'acceptor', 2 for 'donor', and -1 for padding).

    Returns:
    - numpy.ndarray: the one-hot encoded input sequence data.
    - numpy.ndarray: the one-hot encoded output label data.
    """
    return IN_MAP[Xd.astype('int8')], [OUT_MAP[Yd[t].astype('int8')] for t in range(1)]

def check_and_count_motifs(seq, labels, strand):
    """
    Check sequences for donor and acceptor motifs and count their occurrences.

    Parameters:
    - seq: The DNA sequence (str).
    - labels: Array of labels indicating locations of interest in the sequence.
    - strand: The strand (+ or -) indicating the direction of the gene.
    """     
    global donor_motif_counts, acceptor_motif_counts
    for i, label in enumerate(labels):
        if label == 2:  # Donor site
            d_motif = str(seq[i+1:i+3])
            donor_motif_counts[d_motif] = donor_motif_counts.get(d_motif, 0) + 1
        elif label == 1:  # Acceptor site
            a_motif = str(seq[i-2:i])
            acceptor_motif_counts[a_motif] = acceptor_motif_counts.get(a_motif, 0) + 1

def print_motif_counts():
    """
    Print the counts of donor and acceptor motifs.
    """
    global donor_motif_counts, acceptor_motif_counts
    print("Donor motifs:")
    for motif, count in donor_motif_counts.items():
        print(f"{motif}: {count}")
    print("\nAcceptor motifs:")
    for motif, count in acceptor_motif_counts.items():
        print(f"{motif}: {count}")
    print("\nTotal donor motifs: ", sum(donor_motif_counts.values()))
    print("Total acceptor motifs: ", sum(acceptor_motif_counts.values()))