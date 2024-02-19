import argparse
import sys
from spliceaitoolkit import header
__VERSION__ = header.__version__

def parse_args(arglist):
    parser = argparse.ArgumentParser(description='SpliceAI toolkit to retrain your own splice site predictor')

    # Create a parent subparser to house the common subcommands.
    subparsers = parser.add_subparsers(dest='command', required=True, help='Subcommands: create_data, train, eval, predict')
    
    # Create subparsers for each of the subcommands.
    parser_create_data = subparsers.add_parser('create_data', help='Create dataset for your genome for SpliceAI model training')
    parser_train = subparsers.add_parser('train', help='Train the SpliceAI model')
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

    # To-do adding logic to each subcommand.