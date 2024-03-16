import os
import gffutils
from math import ceil
import numpy as np
from spliceaitoolkit.constants import *

def check_and_count_motifs(seq, labels, strand, donor_motif_counts, acceptor_motif_counts):
    """
    Check sequences for donor and acceptor motifs based on labels and strand,
    and return their counts in a dictionary.
    """    
    for i, label in enumerate(labels):
        if label in [1, 2]:  # Check only labeled positions
            if label == 2:
                # Donor site
                d_motif = str(seq[i+1:i+3])
                if d_motif not in donor_motif_counts:
                    donor_motif_counts[d_motif] = 0
                donor_motif_counts[d_motif] += 1
            elif label == 1:
                # Acceptor site
                a_motif = str(seq[i-2:i])
                if a_motif not in acceptor_motif_counts:
                    acceptor_motif_counts[a_motif] = 0
                acceptor_motif_counts[a_motif] += 1


def print_motif_counts(donor_motif_counts, acceptor_motif_counts):
    print("Donor motifs:")
    for motif, count in donor_motif_counts.items():
        print(f"{motif}: {count}")
    print("\nAcceptor motifs:")
    for motif, count in acceptor_motif_counts.items():
        print(f"{motif}: {count}")

###################################################
# Functions for create_datafile
###################################################
def split_chromosomes(seq_dict, split_ratio=0.8):
    """Split chromosomes into training and testing groups."""
    chromosome_sizes = {chromosome: len(record.seq) for chromosome, record in seq_dict.items()}
    total_sequence_size = sum(chromosome_sizes.values())
    target_training_size = 0.8 * total_sequence_size
    sorted_chromosomes = dict(sorted(chromosome_sizes.items(), key=lambda item: item[1], reverse=True))
    training_set = {}
    testing_set = {}
    current_training_size = 0
    for chromo, length in sorted_chromosomes.items():
        if (current_training_size + length <= target_training_size) or not training_set:
            training_set[chromo] = length
            current_training_size += length
        else:
            testing_set[chromo] = length
    # print("Training set:", training_set)
    # print("Testing set:", testing_set)
    # print("Total sequence size:", total_sequence_size)
    # print("current_training_size:", current_training_size)
    # training_set, testing_set, current_training_size, total_sequence_size
    return training_set, testing_set


def create_or_load_db(gff_file, db_file='gff.db'):
    """
    Create a gffutils database from a GFF file, or load it if it already exists.
    """
    if not os.path.exists(db_file):
        print("Creating new database...")
        db = gffutils.create_db(gff_file, dbfn=db_file, force=True, keep_order=True, merge_strategy='merge', sort_attribute_values=True)
    else:
        print("Loading existing database...")
        db = gffutils.FeatureDB(db_file)
    return db
###################################################
# End of functions for create_datafile
###################################################


###################################################
# Functions for create_dataset
###################################################
# One-hot encoding of the inputs: 
# 0 is for padding, 
# 1: A;  2: C;  3: G;  4: T
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


def ceil_div(x, y):
    return int(ceil(float(x)/y))


def replace_non_acgt_to_n(input_string):
    """
    Use a generator expression to go through each character in the input string.
    If the character is in the set of allowed characters, keep it as is.
    Otherwise, replace it with 'N'.
    """
    # Define the set of allowed characters
    allowed_chars = {'A', 'C', 'G', 'T'}    
    return ''.join(char if char in allowed_chars else 'N' for char in input_string)


def reformat_data(X0, Y0):
    """
    This function converts X0, Y0 of the create_datapoints function into
    blocks such that the data is broken down into data points where the
    input is a sequence of length SL+CL_max corresponding to SL nucleotides
    of interest and CL_max context nucleotides, the output is a sequence of
    length SL corresponding to the splicing information of the nucleotides
    of interest. The CL_max context nucleotides are such that they are
    CL_max/2 on either side of the SL nucleotides of interest.
    """
    num_points = ceil_div(len(Y0[0]), SL)
    Xd = np.zeros((num_points, SL+CL_max))
    Yd = [-np.ones((num_points, SL)) for t in range(1)]
    X0 = np.pad(X0, [0, SL], 'constant', constant_values=0)
    Y0 = [np.pad(Y0[t], [0, SL], 'constant', constant_values=-1)
         for t in range(1)]
    for i in range(num_points):
        Xd[i] = X0[SL*i:CL_max+SL*(i+1)]
    for t in range(1):
        for i in range(num_points):
            Yd[t][i] = Y0[t][SL*i:SL*(i+1)]
    return Xd, Yd


def create_datapoints(seq, strand, label):
    """
    This function first converts the sequence into an integer array, where
    A, C, G, T, N are mapped to 1, 2, 3, 4, 0 respectively. If the strand is
    negative, then reverse complementing is done. The labels 
    are directly used as they are, converted into an array of integers,
    where 0, 1, 2 correspond to no splicing, acceptor, donor 
    respectively. It then calls reformat_data and one_hot_encode
    and returns X, Y which can be used by Pytorch Model.
    """
    # I do not need to reverse complement the sequence, as the sequence is already reverse complemented in the previous step
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


def one_hot_encode(Xd, Yd):
    return IN_MAP[Xd.astype('int8')], \
           [OUT_MAP[Yd[t].astype('int8')] for t in range(1)]
###################################################
# End of functions for create_dataset
###################################################
