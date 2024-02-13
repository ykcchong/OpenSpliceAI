import h5py
import numpy as np
import sys, os
import time
# from utils import *
from math import ceil
from constants import *

IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
# One-hot encoding of the inputs: 0 is for padding, and 1, 2, 3, 4 correspond
# to A, C, G, T respectively.

OUT_MAP = np.asarray([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
# One-hot encoding of the outputs: 0 is for no splice, 1 is for acceptor,
# 2 is for donor and -1 is for padding.


def replace_non_acgt_to_n(input_string):
    # Define the set of allowed characters
    allowed_chars = {'A', 'C', 'G', 'T'}
    
    # Use a generator expression to go through each character in the input string.
    # If the character is in the set of allowed characters, keep it as is.
    # Otherwise, replace it with 'N'.
    return ''.join(char if char in allowed_chars else 'N' for char in input_string)


def create_datapoints(seq, strand, tx_start, tx_end, label):
    # This function first converts the sequence into an integer array, where
    # A, C, G, T, N are mapped to 1, 2, 3, 4, 0 respectively. If the strand is
    # negative, then reverse complementing is done. The labels 
    # are directly used as they are, converted into an array of integers,
    # where 0, 1, 2 correspond to no splicing, acceptor, donor 
    # respectively. It then calls reformat_data and one_hot_encode
    # and returns X, Y which can be used by Keras models.
    # print("CL_max: ", CL_max)
    main_seq = seq[CL_max // 2:-CL_max // 2]
    # print("\tmain_seq: ", main_seq)
    # print("\tmain_seq: ", len(main_seq))
    seq = 'N' * (CL_max // 2) + main_seq + 'N' * (CL_max // 2)
    # print("\tseq: ", len(seq))
    seq = seq.upper().replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')
    tx_start = int(tx_start)
    tx_end = int(tx_end)
    # Convert label string to array of integers
    label_array = np.array(list(map(int, list(label))))
    if strand == '+':
        X0 = np.asarray(list(map(int, list(seq))))
        Y0 = label_array
    elif strand == '-':
        X0 = (5 - np.asarray(list(map(int, list(seq[::-1]))))) % 5  # Reverse complement
        Y0 = label_array[::-1]  # Reverse the label array for negative strand
    Y0 = [Y0]
    # print("\tX0.shape: ", X0.shape)
    # print("\tY0.shape: ", Y0.shape)
    Xd, Yd = reformat_data(X0, Y0)
    X, Y = one_hot_encode(Xd, Yd)
    return X, Y


def ceil_div(x, y):
    return int(ceil(float(x)/y))


def reformat_data(X0, Y0):
    # This function converts X0, Y0 of the create_datapoints function into
    # blocks such that the data is broken down into data points where the
    # input is a sequence of length SL+CL_max corresponding to SL nucleotides
    # of interest and CL_max context nucleotides, the output is a sequence of
    # length SL corresponding to the splicing information of the nucleotides
    # of interest. The CL_max context nucleotides are such that they are
    # CL_max/2 on either side of the SL nucleotides of interest.
    num_points = ceil_div(len(Y0), SL)  # Changed to directly use len(Y0)
    Xd = np.zeros((num_points, SL+CL_max))
    Yd = [-np.ones((num_points, SL)) for t in range(1)]
    X0 = np.pad(X0, [0, SL], 'constant', constant_values=0)
    Y0 = [np.pad(Y0[t], [0, SL], 'constant', constant_values=-1)
         for t in range(1)]
    for i in range(num_points):
        start_index = SL*i
        end_index = CL_max + SL*(i+1)
        Xd[i] = X0[start_index:end_index]
    for t in range(1):
        for i in range(num_points):
            Yd[t][i] = Y0[t][SL*i:SL*(i+1)]
    return Xd, Yd


def one_hot_encode(Xd, Yd):
    return IN_MAP[Xd.astype('int8')], \
           [OUT_MAP[Yd[t].astype('int8')] for t in range(1)]


donor_motif_counts = {}  # Initialize counts
acceptor_motif_counts = {}  # Initialize counts

def check_and_count_motifs(seq, labels, strand):
    """
    Check sequences for donor and acceptor motifs based on labels and strand,
    and return their counts in a dictionary.
    """    
    global donor_motif_counts, acceptor_motif_counts
    for i, label in enumerate(labels):
        if label in [1, 2]:  # Check only labeled positions
            if strand == '+':  # For forward strand
                motif = str(seq[i:i+2]) if i > 0 else ""  # Extract preceding 2-base motif
                if label == 1:
                    if motif not in donor_motif_counts:
                        donor_motif_counts[motif] = 0
                    donor_motif_counts[motif] += 1
                elif label == 2:
                    if motif not in acceptor_motif_counts:
                        acceptor_motif_counts[motif] = 0
                    acceptor_motif_counts[motif] += 1
            elif strand == '-':  # For reverse strand, after adjustment
                motif = str(seq[i:i+2]) if i < len(seq) - 1 else ""  # Extract following 2-base motif
                if label == 1:
                    if motif not in donor_motif_counts:
                        donor_motif_counts[motif] = 0
                    donor_motif_counts[motif] += 1    
                elif label == 2:
                    if motif not in acceptor_motif_counts:
                        acceptor_motif_counts[motif] = 0
                    acceptor_motif_counts[motif] += 1

def print_motif_counts():
    global donor_motif_counts, acceptor_motif_counts
    print("Donor motifs:")
    for motif, count in donor_motif_counts.items():
        print(f"{motif}: {count}")
    print("\nAcceptor motifs:")
    for motif, count in acceptor_motif_counts.items():
        print(f"{motif}: {count}")

    print("\nTotal donor motifs: ", sum(donor_motif_counts.values()))
    print("Total acceptor motifs: ", sum(acceptor_motif_counts.values()))


def main():
    start_time = time.time()
    project_root = "/Users/chaokuan-hao/Documents/Projects/spliceAI-MANE/"
    output_dir = f"{project_root}results/gene_sequences_and_labels/"
    os.makedirs(output_dir, exist_ok=True)
    
    # for type in ['test']:
    for type in ['train', 'test']:
        input_file = output_dir + f'datafile_{type}.h5'
        output_file = output_dir + f'dataset_{type}.h5'

        # This is batch processing
    #     with h5py.File(input_file, 'r') as h5f, h5py.File(output_file, 'w') as h5f2:
    #         CHUNK_SIZE = 100  # Adjust based on your dataset size and available memory
            
    #         # Assuming SEQ, LABEL, and STRAND are required for motif checking
    #         seq_data_size = h5f['SEQ'].shape[0]
            
    #         for i in range(0, seq_data_size, CHUNK_SIZE):
    #             # Read SEQ, LABEL, and STRAND data in chunks
    #             chunk = slice(i, min(i + CHUNK_SIZE, seq_data_size))
    #             seq_chunk = h5f['SEQ'][chunk].astype('str')
    #             label_chunk = h5f['LABEL'][chunk].astype('str')
    #             label_chunk_int = [int(char) for char in label_chunk]
    #             strand_chunk = h5f['STRAND'][chunk].astype('str')
    #             print("seq_chunk.shape: ", seq_chunk.shape)
    #             print("label_chunk.shape: ", label_chunk.shape)
    #             print("strand_chunk.shape: ", strand_chunk.shape)
    #             # Call check_and_count_motifs for each chunk
    #         #     check_and_count_motifs(seq_chunk, label_chunk_int, strand_chunk)
    #         # print_motif_counts()
    #     print_motif_counts()


    #     # print("SEQ.shape[0]: ", SEQ.shape[0])
    #     # print("SEQ.shape[0]//CHUNK_SIZE: ", SEQ.shape[0]//CHUNK_SIZE)

    #     # for i in range(SEQ.shape[0]//CHUNK_SIZE):
    #     #     # Each dataset has CHUNK_SIZE genes
            
    #     #     if (i+1) == SEQ.shape[0]//CHUNK_SIZE:
    #     #         NEW_CHUNK_SIZE = CHUNK_SIZE + SEQ.shape[0]%CHUNK_SIZE
    #     #     else:
    #     #         NEW_CHUNK_SIZE = CHUNK_SIZE
    #     #     X_batch = []
    #     #     Y_batch = [[] for t in range(1)]
    #     #     for j in range(NEW_CHUNK_SIZE):
    #     #         idx = i*CHUNK_SIZE + j
    #     #         fixed_seq = replace_non_acgt_to_n(SEQ[idx])
    #     #         X, Y = create_datapoints(fixed_seq, STRAND[idx],
    #     #                                 TX_START[idx], TX_END[idx], LABEL[idx])
    #     #         X_batch.extend(X)
    #     #         for t in range(1):
    #     #             Y_batch[t].extend(Y[t])
    #     #     X_batch = np.asarray(X_batch).astype('int8')
    #     #     print("X_batch.shape: ", X_batch.shape)
    #     #     for t in range(1):
    #     #         Y_batch[t] = np.asarray(Y_batch[t]).astype('int8')
    #     #     print("len(Y_batch[0]): ", len(Y_batch[0]))
    #     #     h5f2.create_dataset('X' + str(i), data=X_batch)
    #     #     h5f2.create_dataset('Y' + str(i), data=Y_batch)

    #         # for name in ['NAME', 'CHROM', 'STRAND', 'TX_START', 'TX_END', 'SEQ', 'LABEL']:
    #         #     data_size = h5f[name].shape[0]
    #         #     print(f"{name}.shape[0]: ", data_size)

    #         #     for i in range(0, data_size, CHUNK_SIZE):
    #         #         # Read data in chunks
    #         #         chunk = slice(i, min(i + CHUNK_SIZE, data_size))
    #         #         data_chunk = h5f[name][chunk]
    #         #         if name in ['STRAND', 'TX_START', 'TX_END', 'SEQ', 'LABEL']:
    #         #             data_chunk = data_chunk.astype('str')

    #         #         # Process data here (e.g., checking motifs, creating datapoints)
    #         #         # Example: check_and_count_motifs(data_chunk)
    #         #         # This is where you would integrate processing functions like `check_and_count_motifs`
    #         #         # and `create_datapoints` in a more memory-efficient manner.

    #         #         # Example of how to write processed data back
    #         #         # h5f2.create_dataset(f'{name}_{i//CHUNK_SIZE}', data=processed_data_chunk)
    #         #         # Make sure to adjust the dataset creation to match your processing output

    #         # Include processing that spans across different datasets within the file handling context to minimize memory usage

    # print(("--- %s seconds ---" % (time.time() - start_time)))


        # This is normal all-in-one processing
        process_size = 0
        if type == 'train':
            process_size = 3000
        elif type == 'test':
            process_size = 1000
        h5f = h5py.File(input_file, 'r')
        NAME = h5f['NAME'][:process_size]
        CHROM = h5f['CHROM'][:process_size]
        STRAND = h5f['STRAND'][:process_size]
        TX_START = h5f['TX_START'][:process_size]
        TX_END = h5f['TX_END'][:process_size]
        SEQ = h5f['SEQ'][:process_size]
        LABEL = h5f['LABEL'][:process_size]
        h5f.close()

        h5f2 = h5py.File(output_file, 'w')
        STRAND = STRAND.astype('str')
        TX_START = TX_START.astype('str')
        TX_END = TX_END.astype('str')
        SEQ = SEQ.astype('str')
        LABEL = LABEL.astype('str')
        CHUNK_SIZE = 100
        print("SEQ.shape[0]: ", SEQ.shape[0])
        # print("CHR.shape[0]: ", CHROM.shape[0])
        print("STRAND.shape[0]: ", STRAND.shape[0])
        print("TX_START.shape[0]: ", TX_START.shape[0])
        print("TX_END.shape[0]: ", TX_END.shape[0])
        print("LABEL.shape[0]: ", LABEL.shape[0])
        # Check motif
        for idx in range(SEQ.shape[0]):
            label_int = [int(char) for char in LABEL[idx]]
            # seq_decode = SEQ[idx].decode('ascii')
            # strand_decode = STRAND[idx].decode('ascii')
            check_and_count_motifs(SEQ[idx], label_int, STRAND[idx])
        print_motif_counts()
        print("SEQ.shape[0]: ", SEQ.shape[0])
        print("SEQ.shape[0]//CHUNK_SIZE: ", SEQ.shape[0]//CHUNK_SIZE)
        for i in range(SEQ.shape[0]//CHUNK_SIZE):
            # Each dataset has CHUNK_SIZE genes
            if (i+1) == SEQ.shape[0]//CHUNK_SIZE:
                NEW_CHUNK_SIZE = CHUNK_SIZE + SEQ.shape[0]%CHUNK_SIZE
            else:
                NEW_CHUNK_SIZE = CHUNK_SIZE
            X_batch = []
            Y_batch = [[] for t in range(1)]
            for j in range(NEW_CHUNK_SIZE):
                idx = i*CHUNK_SIZE + j
                fixed_seq = replace_non_acgt_to_n(SEQ[idx])
                X, Y = create_datapoints(fixed_seq, STRAND[idx],
                                        TX_START[idx], TX_END[idx], LABEL[idx])
                X_batch.extend(X)
                for t in range(1):
                    Y_batch[t].extend(Y[t])
            X_batch = np.asarray(X_batch).astype('int8')
            print("X_batch.shape: ", X_batch.shape)
            for t in range(1):
                Y_batch[t] = np.asarray(Y_batch[t]).astype('int8')
            print("len(Y_batch[0]): ", len(Y_batch[0]))
            h5f2.create_dataset('X' + str(i), data=X_batch)
            h5f2.create_dataset('Y' + str(i), data=Y_batch)
        h5f2.close()
    print(("--- %s seconds ---" % (time.time() - start_time)))

if __name__ == "__main__":
    main()