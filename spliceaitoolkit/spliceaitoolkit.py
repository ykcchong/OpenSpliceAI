import argparse
import os, sys, time
import random
import h5py
import numpy as np
from spliceaitoolkit import header
from spliceaitoolkit.create_data import create_datafile, create_dataset, verify_h5_file
from spliceaitoolkit.train import train
from spliceaitoolkit.calibrate import calibrate
from spliceaitoolkit.fine_tune import fine_tune
from spliceaitoolkit.predict import predict
from spliceaitoolkit.variant import variant

__VERSION__ = header.__version__

def parse_args_create_data(subparsers):
    parser_create_data = subparsers.add_parser('create-data', help='Create dataset for your genome for SpliceAI model training')
    parser_create_data.add_argument('--annotation-gff', type=str, required=True, help='Path to the GFF file')
    parser_create_data.add_argument('--genome-fasta', type=str, required=True, help='Path to the FASTA file')
    parser_create_data.add_argument('--output-dir', type=str, required=True, help='Output directory to save the data')
    parser_create_data.add_argument('--parse-type', type=str, default='maximum', choices=['maximum', 'all_isoforms'], help='Type of transcript processing')
    parser_create_data.add_argument('--biotype', type=str, default='protein-coding', choices=['protein-coding', 'non-coding'], help='Biotype of transcript processing')
    parser_create_data.add_argument('--chr-split', type=str, choices=['train-test','test'], default='train-test', help='Whether to obtain testing or both training and testing groups')
    '''AM: newly added flags below vv'''
    parser_create_data.add_argument('--split-method', type=str, choices=['random', 'human'], default='random', help='Chromosome split method for training and testing dataset')
    parser_create_data.add_argument('--split-ratio', type=float, default=0.8, help='Ratio of training and testing dataset')
    parser_create_data.add_argument('--canonical-only', action='store_true', default=True, help='Flag to obtain only canonical splice site pairs')
    parser_create_data.add_argument('--flanking-size', type=int, default=80, help='Sum of flanking sequence lengths on each side of input (i.e. 40+40)')
    parser_create_data.add_argument('--verify-h5', action='store_true', default=False, help='Verify the generated HDF5 file(s)')

def parse_args_train(subparsers):
    parser_train = subparsers.add_parser('train', help='Train the SpliceAI model')
    parser_train.add_argument('--disable-wandb', '-d', action='store_true', default=False)
    parser_train.add_argument('--output-dir', '-o', type=str, required=True, help='Output directory to save the data')
    parser_train.add_argument('--project-name', '-s', type=str)
    parser_train.add_argument('--flanking-size', '-f', type=int, default=80)
    parser_train.add_argument('--random-seed', '-r', type=int, default=42)
    parser_train.add_argument('--exp-num', '-e', type=str, default=0)
    parser_train.add_argument('--train-dataset', '-train', type=str)
    parser_train.add_argument('--test-dataset', '-test', type=str)
    parser_train.add_argument('--loss', '-l', type=str, default="cross_entropy_loss", help='The loss function to train SpliceAI model')
    parser_train.add_argument('--model', '-m', default="SpliceAI", type=str)

def parse_args_calibrate(subparsers):
    parser_calibrate = subparsers.add_parser('calibrate', help='Calibrate the SpliceAI model')
    parser_calibrate.add_argument('--output-dir', '-o', type=str, required=True, help='Output directory to save the data')
    pass

def parse_args_fine_tune(subparsers):
    parser_fine_tune = subparsers.add_parser('fine-tune', help='Fine-tune a pre-trained SpliceAI model on new data.')
    parser_fine_tune.add_argument("--output_dir", '-o', type=str, required=True, help="Output directory for model checkpoints and logs")
    parser_fine_tune.add_argument("--project_name", '-s', type=str, required=True, help="Project name for organizing outputs")
    parser_fine_tune.add_argument("--exp_num", '-e', type=int, default=0, help="Experiment number")
    parser_fine_tune.add_argument("--flanking_size", '-f', type=int, default=80, choices=[80, 400, 2000, 10000], help="Flanking sequence size")
    parser_fine_tune.add_argument("--model_path", '-m', type=str, required=True, help="Path to the pre-trained model")
    parser_fine_tune.add_argument("--loss", '-l', type=str, default='cross_entropy_loss', choices=["cross_entropy_loss", "focal_loss"], help="Loss function for training")
    parser_fine_tune.add_argument("--train_dataset", '-train', type=str, required=True, help="Path to the training dataset")
    parser_fine_tune.add_argument("--test_dataset", '-test', type=str, required=True, help="Path to the testing dataset")
    parser_fine_tune.add_argument("--disable_wandb", '-d', action='store_true', default=False, help="Disable Weights & Biases logging")
    parser_fine_tune.add_argument("--random_seed", '-r', type=int, default=42, help="Random seed for reproducibility")
    parser_fine_tune.add_argument("--unfreeze", '-u', type=int, default=1, help="Number of layers to unfreeze for fine-tuning")


def parse_args_predict(subparsers):
    parser_predict = subparsers.add_parser('predict', help='Predict splice sites in a given sequence using the SpliceAI model')
    parser_predict.add_argument('--model', '-m', type=str, default="SpliceAI", help='Path to a PyTorch SpliceAI model file or "SpliceAI" for the default model')
    parser_predict.add_argument('--output-dir', '-o', type=str, required=True, help='Output directory to save the data')
    parser_predict.add_argument('--flanking-size', '-f', type=int, default=80, help='Sum of flanking sequence lengths on each side of input (i.e. 40+40)')
    parser_predict.add_argument('--input-sequence', '-i', type=str, help="Path to FASTA file of the input sequence")
    parser_predict.add_argument('--annotation-file', '-a', type=str, required=False, help="Path to GFF file of coordinates for genes")
    parser_predict.add_argument('--threshold', '-t', type=float, default=1e-6, help="Threshold to determine acceptor and donor sites")
    parser_predict.add_argument('--predict-all', '-p', action='store_true', required=False, help="Writes all collected predictions to an intermediate file (Warning: on full genomes, will consume much space.)")
    parser_predict.add_argument('--debug', '-D', action='store_true', required=False, help="Run in debug mode (debug statements are printed to stderr)")
    '''AM: very optional flags below vv'''
    parser_predict.add_argument('--hdf-threshold', type=int, default=0, help='Maximum size before reading sequence into an HDF file for storage')
    parser_predict.add_argument('--flush-threshold', type=int, default=500, help='Maximum number of predictions before flushing to file')
    parser_predict.add_argument('--split-threshold', type=int, default=1500000, help='Maximum length of FASTA entry before splitting')
    parser_predict.add_argument('--chunk-size', type=int, default=100, help='Chunk size for loading HDF5 dataset')

def parse_args_variant(subparsers):
    parser_variant = subparsers.add_parser('variant', help='Label genetic variations with their predicted effects on splicing.')
    parser_variant.add_argument('-R', metavar='reference', required=True, help='path to the reference genome fasta file')
    parser_variant.add_argument('-A', metavar='annotation', required=True, help='"grch37" (GENCODE V24lift37 canonical annotation file in '
                                                                                'package), "grch38" (GENCODE V24 canonical annotation file in '
                                                                                'package), or path to a similar custom gene annotation file')
    parser_variant.add_argument('-I', metavar='input', nargs='?', default=sys.stdin, help='path to the input VCF file, defaults to standard in')
    parser_variant.add_argument('-O', metavar='output', nargs='?', default=sys.stdout, help='path to the output VCF file, defaults to standard out')
    parser_variant.add_argument('-D', metavar='distance', nargs='?', default=50, type=int, choices=range(0, 5000),
                                    help='maximum distance between the variant and gained/lost splice '
                                        'site, defaults to 50')
    parser_variant.add_argument('-M', metavar='mask', nargs='?', default=0, type=int, choices=[0, 1], 
                                    help='mask scores representing annotated acceptor/donor gain and '
                                        'unannotated acceptor/donor loss, defaults to 0')
    '''AM: newly added flags below vv'''
    parser_variant.add_argument('--model', '-m', default="SpliceAI", type=str, help='Path to a SpliceAI model file, or path to a directory of SpliceAI models, or "SpliceAI" for the default model')
    parser_variant.add_argument('--flanking-size', '-f', type=int, default=80, help='Sum of flanking sequence lengths on each side of input (i.e. 40+40)')
    parser_variant.add_argument('--model-type', '-t', type=str, choices=['keras', 'pytorch'], default='pytorch', help='Type of model file (keras or pytorch)')
    parser_variant.add_argument('--precision', '-p', type=int, default=2, help='Number of decimal places to round the output scores')
 

def parse_args(arglist):
    parser = argparse.ArgumentParser(description='SpliceAI toolkit to retrain your own splice site predictor')
    # Create a parent subparser to house the common subcommands.
    subparsers = parser.add_subparsers(dest='command', required=True, help='Subcommands: create-data, train, calibrate, predict, fine-tune, variant')
    parse_args_create_data(subparsers)
    parse_args_train(subparsers)
    parse_args_calibrate(subparsers)
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
        if args.verify_h5:
            verify_h5_file.verify_h5(args)
    elif args.command == 'train':
        train.train(args)
    elif args.command == 'calibrate':
        calibrate.calibrate(args)
    elif args.command == 'fine-tune':
        fine_tune.fine_tune(args)
    elif args.command == 'predict':
        predict.predict(args)
    elif args.command == 'variant':
        variant.variant(args)

    # To-do adding logic to each subcommand.
