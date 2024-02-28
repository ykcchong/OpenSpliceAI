import argparse
import os, sys, time
import random
import h5py
import numpy as np
from spliceaitoolkit import header
from spliceaitoolkit.create_data import create_datafile, create_dataset
from spliceaitoolkit.train import train
# , verify_h5_file 
__VERSION__ = header.__version__

def get_all_chromosomes(db):
    """Extract all unique chromosomes from the GFF database."""
    chromosomes = set()
    for feature in db.all_features():
        chromosomes.add(feature.seqid)
    return list(chromosomes)

def split_chromosomes(chromosomes, method='random', split_ratio=0.8):
    """Split chromosomes into training and testing groups."""
    if method == 'random':
        random.shuffle(chromosomes)
        split_point = int(len(chromosomes) * split_ratio)
        train_chroms = {chrom: 0 for chrom in chromosomes[:split_point]}
        test_chroms = {chrom: 0 for chrom in chromosomes[split_point:]}
    else:
        # Implement other methods if needed
        train_chroms, test_chroms = {}, {}
    return train_chroms, test_chroms


def parse_args(arglist):
    parser = argparse.ArgumentParser(description='SpliceAI toolkit to retrain your own splice site predictor')

    # Create a parent subparser to house the common subcommands.
    subparsers = parser.add_subparsers(dest='command', required=True, help='Subcommands: create_data, train, eval, predict')
    
    # Create subparsers for each of the subcommands.
    parser_create_data = subparsers.add_parser('create-data', help='Create dataset for your genome for SpliceAI model training')
    parser_create_data.add_argument('--annotation-gff', type=str, required=True, help='Path to the GFF file')
    parser_create_data.add_argument('--genome-fasta', type=str, required=True, help='Path to the FASTA file')
    parser_create_data.add_argument('--output-dir', type=str, required=True, help='Output directory to save the data')
    parser_create_data.add_argument('--parse-type', type=str, default='maximum', choices=['maximum', 'all_isoforms'], help='Type of transcript processing')
    # parser_create_data.add_argument('--chrom-dict', type=str, required=True, help='Path to the chromosome dictionary file')

    parser_train = subparsers.add_parser('train', help='Train the SpliceAI model')
    parser_train.add_argument('--disable-wandb', '-d', action='store_true', default=False)
    parser_train.add_argument('--project-root', '-p', type=str)
    parser_train.add_argument('--project-name', '-s', type=str)
    parser_train.add_argument('--flanking-size', '-f', type=int, default=80)
    parser_train.add_argument('--exp-num', '-e', type=str, default=0)
    parser_train.add_argument('--training-target', '-t', type=str, default="SpliceAI")
    parser_train.add_argument('--train-dataset', '-train', type=str)
    parser_train.add_argument('--test-dataset', '-test', type=str)
    parser_train.add_argument('--model', '-m', default="SpliceAI", type=str)
    args = parser_train.parse_args()


    parser_eval = subparsers.add_parser('eval', help='Evaluate the SpliceAI model')
    parser_predict = subparsers.add_parser('predict', help='Predict using the SpliceAI model')

    ###################################
    # END for the LiftOn params
    ###################################
    args = parser.parse_args(arglist)
    return args

def main(arglist=None):
    # ANSI Shadow
    banner = '''
====================================================================
Deep learning framework to train your own SpliceAI model
====================================================================


███████╗██████╗ ██╗     ██╗ ██████╗███████╗ █████╗ ██╗   ████████╗ ██████╗  ██████╗ ██╗     ██╗  ██╗██╗████████╗
██╔════╝██╔══██╗██║     ██║██╔════╝██╔════╝██╔══██╗██║   ╚══██╔══╝██╔═══██╗██╔═══██╗██║     ██║ ██╔╝██║╚══██╔══╝
███████╗██████╔╝██║     ██║██║     █████╗  ███████║██║█████╗██║   ██║   ██║██║   ██║██║     █████╔╝ ██║   ██║   
╚════██║██╔═══╝ ██║     ██║██║     ██╔══╝  ██╔══██║██║╚════╝██║   ██║   ██║██║   ██║██║     ██╔═██╗ ██║   ██║   
███████║██║     ███████╗██║╚██████╗███████╗██║  ██║██║      ██║   ╚██████╔╝╚██████╔╝███████╗██║  ██╗██║   ██║   
╚══════╝╚═╝     ╚══════╝╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝╚═╝      ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝   ╚═╝   
    '''
    print(banner, file=sys.stderr)
    print(f"{__VERSION__}\n", file=sys.stderr)
    args = parse_args(arglist)

    if args.command == 'create-data':
        os.makedirs(args.output_dir, exist_ok=True)
        db = create_datafile.create_or_load_db(args.annotation_gff, db_file=f'{args.annotation_gff}_db')
        # Find all distinct chromosomes and split them
        all_chromosomes = get_all_chromosomes(db)
        TRAIN_CHROM_GROUP, TEST_CHROM_GROUP = split_chromosomes(all_chromosomes, method='random')  # Or any other method you prefer
        print("TRAIN_CHROM_GROUP: ", TRAIN_CHROM_GROUP)
        print("TEST_CHROM_GROUP: ", TEST_CHROM_GROUP)
        print("--- Step 1: Creating datafile.h5 ... ---")
        start_time = time.time()
        create_datafile.get_sequences_and_labels(db, args.genome_fasta, args.output_dir, type="train", chrom_dict=TRAIN_CHROM_GROUP, parse_type=args.parse_type)
        create_datafile.get_sequences_and_labels(db, args.genome_fasta, args.output_dir, type="test", chrom_dict=TEST_CHROM_GROUP, parse_type=args.parse_type)
        create_datafile.print_motif_counts()
        print("--- %s seconds ---" % (time.time() - start_time))

        print("--- Step 2: Creating dataset.h5 ... ---")
        start_time = time.time()
        for type in ['train', 'test']:
            print(("\tProcessing %s ..." % type))
            input_file = f"{args.output_dir}/datafile_{type}.h5"
            output_file = f"{args.output_dir}/dataset_{type}.h5"
            # output_file = f"{os.path.dirname(input_file)}/dataset_{type}.h5"
            print("\tReading datafile.h5 ... ")
            h5f = h5py.File(input_file, 'r')
            STRAND = h5f['STRAND'][:]
            TX_START = h5f['TX_START'][:]
            TX_END = h5f['TX_END'][:]
            SEQ = h5f['SEQ'][:]
            LABEL = h5f['LABEL'][:]
            h5f.close()

            h5f2 = h5py.File(output_file, 'w')
            CHUNK_SIZE = 100
            seq_num = SEQ.shape[0]
            print("seq_num: ", seq_num)
            print("STRAND.shape[0]: ", STRAND.shape[0])
            print("TX_START.shape[0]: ", TX_START.shape[0])
            print("TX_END.shape[0]: ", TX_END.shape[0])
            print("LABEL.shape[0]: ", LABEL.shape[0])
            # # Check motif
            # for idx in range(seq_num):
            #     label_decode = LABEL[idx].decode('ascii')
            #     seq_decode = SEQ[idx].decode('ascii')
            #     strand_decode = STRAND[idx].decode('ascii')
            #     label_int = [int(char) for char in label_decode]
            #     check_and_count_motifs(seq_decode, label_int, strand_decode)
            # print_motif_counts()

            # Create dataset
            for i in range(seq_num//CHUNK_SIZE):
                # Each dataset has CHUNK_SIZE genes
                if (i+1) == seq_num//CHUNK_SIZE:
                    NEW_CHUNK_SIZE = CHUNK_SIZE + seq_num%CHUNK_SIZE
                else:
                    NEW_CHUNK_SIZE = CHUNK_SIZE
                X_batch = []
                Y_batch = [[] for t in range(1)]
                for j in range(NEW_CHUNK_SIZE):
                    idx = i*CHUNK_SIZE + j
                    seq_decode = SEQ[idx].decode('ascii')
                    strand_decode = STRAND[idx].decode('ascii')
                    tx_start_decode = TX_START[idx].decode('ascii')
                    tx_end_decode = TX_END[idx].decode('ascii')
                    label_decode = LABEL[idx].decode('ascii')
                    fixed_seq = create_dataset.replace_non_acgt_to_n(seq_decode)
                    X, Y = create_dataset.create_datapoints(fixed_seq, strand_decode, label_decode)                
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
        print("--- %s seconds ---" % (time.time() - start_time))

    elif args.command == 'train':
        train.train_epoch(args)
    
    # elif args.command == 'eval':
    #     pass
    # elif args.command == 'predict':
    #     pass


    # To-do adding logic to each subcommand.