import os
import sys
import argparse
import logging
import pysam
import numpy as np
from spliceaitoolkit.variant.utils import *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def variant(args):
    print("Running SpliceAI-toolkit with 'variant' mode")

    # Error handling for required arguments
    if None in [args.I, args.O, args.R, args.A, args.model, args.flanking_size]:
        logging.error('Usage: spliceai [-h] [-m [model]] [-f [flanking_size]] [-I [input]] [-O [output]] -R reference -A annotation '
                      '[-D [distance]] [-M [mask]]')
        exit(1)

    # Reading input VCF file
    print('\t[INFO] Reading input VCF file')
    try:
        vcf = pysam.VariantFile(args.I)
    except (IOError, ValueError) as e:
        logging.error('Error reading input file: {}'.format(e))
        exit(1)

    # Adding annotation to the header
    header = vcf.header
    header.add_line('##INFO=<ID=SpliceAI,Number=.,Type=String,Description="SpliceAI-tookit variant '
                    'annotation. These include delta scores (DS) and delta positions (DP) for '
                    'acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). '
                    'Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL">')

    # Generating output VCF file
    print('\t[INFO] Generating output VCF file')
    output_dir = initialize_paths(os.path.dirname(args.O), args.flanking_size)
    try:
        output = pysam.VariantFile(args.O, mode='w', header=header)
    except (IOError, ValueError) as e:
        logging.error('Error generating output VCF file: {}'.format(e))
        exit(1)

    # Initialize the Annotator class
    logging.info('Initializing Annotator class')
    ann = Annotator(args.R, args.A, output_dir, args.model, int(args.flanking_size))

    # Process each record in the VCF
    for record in vcf:
        scores = get_delta_scores(record, ann, args.D, args.M)
        if scores:
            record.info['SpliceAI'] = scores
        output.write(record)

    # Close input and output VCF files
    vcf.close()
    output.close()
    logging.info('Annotation completed and written to output VCF file')



# def variant(args):

#     print("Running SpliceAI-toolkit with 'variant' mode")

#     # Error handling 
#     if None in [args.I, args.O, args.D, args.M]:
#         # logging.error('Usage: spliceai [-h] [-I [input]] [-O [output]] -R reference -A annotation '
#         #               '[-D [distance]] [-M [mask]]')
#         # exit()
#         print("Incorrect arguments supplied")
#         exit()

#     # Getting inputs
#     print('\t[INFO] Reading input VCF file')
#     try:
#         vcf = pysam.VariantFile(args.I)
#     except (IOError, ValueError) as e:
#         # logging.error('{}'.format(e))
#         # exit()
#         print("Error reading input file")
#         exit()

#     header = vcf.header
#     header.add_line('##INFO=<ID=SpliceAI,Number=.,Type=String,Description="SpliceAI-tookit variant '
#                     'annotation. These include delta scores (DS) and delta positions (DP) for '
#                     'acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). '
#                     'Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL">')

#     # Generating output VCF file
#     print('\t[INFO] Generating output VCF file')
#     try:
#         output = pysam.VariantFile(args.O, mode='w', header=header)
#     except (IOError, ValueError) as e:
#         # logging.error('{}'.format(e))
#         # exit()
#         print("Error generating output VCF file")
#         exit()

#     ### INITIALIZE ANNOTATOR CLASS ###
#     ann = Annotator(args.R, args.A)

#     for record in vcf:
#         scores = get_delta_scores(record, ann, args.D, args.M)
#         if len(scores) > 0:
#             record.info['SpliceAI'] = scores
#         output.write(record)

#     vcf.close()
#     output.close()

#     # need to test first
#     # vcf file -> extract sequence at every coordinate -> predict on every variant -> generate vcf file with scores for each variant
#     # how to tolerate longer indels? 
