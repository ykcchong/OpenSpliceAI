import argparse
import os, sys, time
import random
import h5py
import numpy as np
from spliceaitoolkit import header
from spliceaitoolkit.create_data import create_datafile, create_dataset
from spliceaitoolkit.train import train
from spliceaitoolkit.fine_tune import fine_tune
from spliceaitoolkit.predict import predict
from spliceaitoolkit.variant import variant

try:
    from sys.stdin import buffer as std_in
    from sys.stdout import buffer as std_out
except ImportError:
    from sys import stdin as std_in
    from sys import stdout as std_out

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

def parse_args_create_data(subparsers):
    parser_create_data = subparsers.add_parser('create-data', help='Create dataset for your genome for SpliceAI model training')
    parser_create_data.add_argument('--annotation-gff', type=str, required=True, help='Path to the GFF file')
    parser_create_data.add_argument('--genome-fasta', type=str, required=True, help='Path to the FASTA file')
    parser_create_data.add_argument('--output-dir', type=str, required=True, help='Output directory to save the data')
    parser_create_data.add_argument('--parse-type', type=str, default='maximum', choices=['maximum', 'all_isoforms'], help='Type of transcript processing')
    # parser_create_data.add_argument('--chrom-split', type=str, required=True, help='Chromosome split method for training and testing dataset')


def parse_args_train(subparsers):
    parser_train = subparsers.add_parser('train', help='Train the SpliceAI model')
    parser_train.add_argument('--disable-wandb', '-d', action='store_true', default=False)
    parser_train.add_argument('--output-dir', '-o', type=str, required=True, help='Output directory to save the data')
    parser_train.add_argument('--project-name', '-s', type=str)
    parser_train.add_argument('--flanking-size', '-f', type=int, default=80)
    parser_train.add_argument('--exp-num', '-e', type=str, default=0)
    parser_train.add_argument('--train-dataset', '-train', type=str)
    parser_train.add_argument('--test-dataset', '-test', type=str)
    parser_train.add_argument('--loss', '-l', type=str, default="cross_entropy_loss", help='The loss function to train SpliceAI model')
    parser_train.add_argument('--model', '-m', default="SpliceAI", type=str)


def parse_args_fine_tune(subparsers):
    parser_fine_tune = subparsers.add_parser('fine-tune', help='Fine-tune a pre-trained SpliceAI model.')
    parser_fine_tune.add_argument('--disable-wandb', '-d', action='store_true', default=False)
    parser_fine_tune.add_argument('--input-model', '-im', default="SpliceAI", type=str)
    parser_fine_tune.add_argument('--output-model', '-om', default="SpliceAI", type=str)
    parser_fine_tune.add_argument('--output-dir', '-o', type=str, required=True, help='Output directory to save the data')
    parser_fine_tune.add_argument('--project-name', '-s', type=str)
    parser_fine_tune.add_argument('--flanking-size', '-f', type=int, default=80)
    parser_fine_tune.add_argument('--exp-num', '-e', type=str, default=0)
    parser_fine_tune.add_argument('--train-dataset', '-train', type=str)
    parser_fine_tune.add_argument('--test-dataset', '-test', type=str)
    parser_fine_tune.add_argument('--loss', '-l', type=str, default="cross_entropy_loss", help='The loss function to train SpliceAI model')


def parse_args_predict(subparsers):
    parser_predict = subparsers.add_parser('predict', help='Predict splice sites in a given sequence using the SpliceAI model')
    parser_predict.add_argument('--model', '-m', default="SpliceAI", type=str)
    parser_predict.add_argument('--output-dir', '-o', type=str, required=True, help='Output directory to save the data')
    parser_predict.add_argument('--flanking-size', '-f', type=int, default=80, help='Sum of flanking sequence lengths on each side of input (i.e. 40+40)')
    parser_predict.add_argument('--input-sequence', '-i', type=str, help="Path to FASTA file of the input sequence")


def parse_args_variant(subparsers):
    parser_variant = subparsers.add_parser('variant', help='Label genetic variations with their predicted effects on splicing.')
    parser_variant.add_argument('--model', '-m', default="SpliceAI", type=str)
    parser_variant.add_argument('-I', metavar='input', nargs='?', default=std_in,
                        help='path to the input VCF file, defaults to standard in')
    parser_variant.add_argument('-O', metavar='output', nargs='?', default=std_out,
                        help='path to the output VCF file, defaults to standard out')
    parser_variant.add_argument('-R', metavar='reference', required=True,
                        help='path to the reference genome fasta file')
    parser_variant.add_argument('-A', metavar='annotation', required=True,
                        help='"grch37" (GENCODE V24lift37 canonical annotation file in '
                             'package), "grch38" (GENCODE V24 canonical annotation file in '
                             'package), or path to a similar custom gene annotation file')
    parser_variant.add_argument('-D', metavar='distance', nargs='?', default=50,
                        type=int, choices=range(0, 5000),
                        help='maximum distance between the variant and gained/lost splice '
                             'site, defaults to 50')
    parser_variant.add_argument('-M', metavar='mask', nargs='?', default=0,
                        type=int, choices=[0, 1],
                        help='mask scores representing annotated acceptor/donor gain and '
                             'unannotated acceptor/donor loss, defaults to 0')
 

def parse_args(arglist):
    parser = argparse.ArgumentParser(description='SpliceAI toolkit to retrain your own splice site predictor')
    # Create a parent subparser to house the common subcommands.
    subparsers = parser.add_subparsers(dest='command', required=True, help='Subcommands: create-data, train, predict, fine-tune, variant')
    parse_args_create_data(subparsers)
    parse_args_train(subparsers)
    parse_args_fine_tune(subparsers)
    parse_args_predict(subparsers)
    parse_args_variant(subparsers)
    if arglist is not None:
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()
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
    print(args)

    if args.command == 'create-data':
        create_datafile.create_datafile(args)
        create_dataset.create_dataset(args)
    elif args.command == 'train':
        train.train(args)
    elif args.command == 'fine-tune':
        fine_tune.fine_tune(args)
    elif args.command == 'predict':
        predict.predict(args)
    elif args.command == 'variant':
        variant.variant(args)

    # To-do adding logic to each subcommand.