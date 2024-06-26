import os
import sys
import argparse
import logging
import pysam
import numpy as np
from openspliceai.variant.utils import *
from tqdm import tqdm

def variant(args):
    print("Running SpliceAI-toolkit with 'variant' mode")
    start_time = time.time()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Error handling for required arguments
    if None in [args.I, args.O, args.R, args.A, args.model, args.flanking_size]:
        logging.error('Usage: spliceai [-h] [-m [model]] [-f [flanking_size]] [-I [input]] [-O [output]] -R reference -A annotation '
                      '[-D [distance]] [-M [mask]]')
        exit(1)

    # Define arguments
    ref_genome = args.R
    annotation = args.A
    input_vcf = args.I
    output_vcf = args.O
    distance = args.D
    mask = args.M
    model = args.model
    flanking_size = args.flanking_size
    model_type = args.model_type

    # Reading input VCF file
    print('\t[INFO] Reading input VCF file')
    try:
        vcf = pysam.VariantFile(input_vcf)
    except (IOError, ValueError) as e:
        logging.error('Error reading input file: {}'.format(e))
        exit(1)

    # Adding annotation to the header
    header = vcf.header
    header.add_line('##INFO=<ID=SpliceAI,Number=.,Type=String,Description="OpenSpliceAI variant '
                    'annotation. These include delta scores (DS) and delta positions (DP) for '
                    'acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). '
                    'Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL">')

    # Generating output VCF file
    print('\t[INFO] Generating output VCF file')
    output_dir = os.path.dirname(output_vcf)
    try:
        output = pysam.VariantFile(output_vcf, mode='w', header=header)
    except (IOError, ValueError) as e:
        logging.error('Error generating output VCF file: {}'.format(e))
        exit(1)

    # Setup the Annotator based on reference genome and annotation
    logging.info('Initializing Annotator class')
    ann = Annotator(ref_genome, annotation, model, model_type, flanking_size)

    # Obtain delta score for each variant in VCF
    for record in tqdm(vcf):
        scores = get_delta_scores(record, ann, distance, mask, flanking_size)
        if scores:
            record.info['SpliceAI'] = scores
        output.write(record)

    # Close input and output VCF files
    vcf.close()
    output.close()
    logging.info('Annotation completed and written to output VCF file')
    
    print("--- %s seconds ---" % (time.time() - start_time))