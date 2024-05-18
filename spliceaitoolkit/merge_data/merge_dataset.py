import h5py
import numpy as np
import sys, os
import time
# from utils import *
import argparse
import spliceaitoolkit.create_data.utils as utils

donor_motif_counts = {}  # Initialize counts
acceptor_motif_counts = {}  # Initialize counts

def merge_dataset(args):
    print("--- Merging dataset.h5 ... ---")
    start_time = time.time()

    dataset_ls = [] 
    if args.chr_split == 'test':
        dataset_ls.append('test')
    elif args.chr_split == 'train-test':
        dataset_ls.append('test')
        dataset_ls.append('train')
    for dataset_type in dataset_ls:
        print(("\tProcessing %s ..." % dataset_type))
        # if args.biotype =="non-coding":
        #     input_file = f"{args.output_dir}/datafile_{dataset_type}_ncRNA.h5"
        #     output_file = f"{args.output_dir}/dataset_{dataset_type}_ncRNA.h5"
        # elif args.biotype =="protein-coding":
        #     input_file = f"{args.output_dir}/datafile_{dataset_type}.h5"
        #     output_file = f"{args.output_dir}/dataset_{dataset_type}.h5"

        os.makedirs(args.output_dir, exist_ok=True)
        output_file = f"{args.output_dir}/dataset_{dataset_type}.h5"
        h5f2 = h5py.File(output_file, 'w')
        print(f"\t output {output_file} ... ")

        X_concat = []
        Y_concat = [[] for _ in range(1)]

        for input_dir in args.input_dir:
            input_file = f"{input_dir}/dataset_{dataset_type}.h5"
            print(f"\tReading {input_file} ... ")
            with h5py.File(input_file, 'r') as h5f:
                keys = list(h5f.keys())
                print(f"\t keys: {keys}")
                # for key in keys:
                #     if key.startswith('X'):
                #         X_data = h5f[key][:]
                #         X_concat.append(X_data)
                #     elif key.startswith('Y'):
                #         Y_data = h5f[key][:]
                #         Y_concat[0].append(Y_data)

        X_concat = np.concatenate(X_concat, axis=0).astype('int8')
        Y_concat = np.concatenate(Y_concat[0], axis=0).astype('int8')

        h5f2.create_dataset('X', data=X_concat)
        h5f2.create_dataset('Y', data=Y_concat)

        h5f2.close()



        # CHUNK_SIZE = 100
        # seq_num = SEQ.shape[0]
        # print("seq_num: ", seq_num)
        # print("STRAND.shape[0]: ", STRAND.shape[0])
        # print("TX_START.shape[0]: ", TX_START.shape[0])
        # print("TX_END.shape[0]: ", TX_END.shape[0])
        # print("LABEL.shape[0]: ", LABEL.shape[0])
        # # # Check motif
        # # for idx in range(seq_num):
        # #     label_decode = LABEL[idx].decode('ascii')
        # #     seq_decode = SEQ[idx].decode('ascii')
        # #     strand_decode = STRAND[idx].decode('ascii')
        # #     label_int = [int(char) for char in label_decode]
        # #     utils.check_and_count_motifs(seq_decode, label_int, strand_decode, donor_motif_counts, acceptor_motif_counts)
        # # utils.print_motif_counts(donor_motif_counts, acceptor_motif_counts)
        # # Create dataset
        # for i in range(seq_num//CHUNK_SIZE):
        #     # Each dataset has CHUNK_SIZE genes
        #     if (i+1) == seq_num//CHUNK_SIZE:
        #         NEW_CHUNK_SIZE = CHUNK_SIZE + seq_num%CHUNK_SIZE
        #     else:
        #         NEW_CHUNK_SIZE = CHUNK_SIZE
        #     X_batch = []
        #     Y_batch = [[] for t in range(1)]
        #     for j in range(NEW_CHUNK_SIZE):
        #         idx = i*CHUNK_SIZE + j
        #         seq_decode = SEQ[idx].decode('ascii')
        #         strand_decode = STRAND[idx].decode('ascii')
        #         tx_start_decode = TX_START[idx].decode('ascii')
        #         tx_end_decode = TX_END[idx].decode('ascii')
        #         label_decode = LABEL[idx].decode('ascii')
        #         fixed_seq = utils.replace_non_acgt_to_n(seq_decode)
        #         X, Y = utils.create_datapoints(fixed_seq, strand_decode, label_decode)                
        #         X_batch.extend(X)
        #         for t in range(1):
        #             Y_batch[t].extend(Y[t])
        #     X_batch = np.asarray(X_batch).astype('int8')
        #     print("X_batch.shape: ", X_batch.shape)
        #     for t in range(1):
        #         Y_batch[t] = np.asarray(Y_batch[t]).astype('int8')
        #     print("len(Y_batch[0]): ", len(Y_batch[0]))
        #     h5f2.create_dataset('X' + str(i), data=X_batch)
        #     h5f2.create_dataset('Y' + str(i), data=Y_batch)
        # h5f2.close()
    print("--- %s seconds ---" % (time.time() - start_time))