import argparse
import logging
import pysam
import time
from openspliceai.variant.utils import *
from tqdm import tqdm
import sys, os

# NOTE: if running with gpu, note that cudnn version should be 8.9.6 or higher, numpy <2.0.0

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
    precision = args.precision
    
    print(f'''Running with genome: {ref_genome}, annotation: {annotation}, 
          model(s): {model}, model_type: {model_type}, 
          input: {input_vcf}, output: {output_vcf}, 
          distance: {distance}, mask: {mask}, flanking_size: {flanking_size}, precision: {precision}''')

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
    os.makedirs(os.path.dirname(output_vcf), exist_ok=True)
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
        scores = get_delta_scores(record, ann, distance, mask, flanking_size, precision)
        if scores:
            record.info['SpliceAI'] = scores
        output.write(record)

    # Close input and output VCF files
    vcf.close()
    output.close()
    logging.info('Annotation completed and written to output VCF file')
    
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    parser_variant = argparse.ArgumentParser(description='Label genetic variations with their predicted effects on splicing.')

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
    parser_variant.add_argument('--model', '-m', default="SpliceAI", type=str, help='Path to a SpliceAI model file, or path to a directory of SpliceAI models, or "SpliceAI" for the default model')
    parser_variant.add_argument('--flanking-size', '-f', type=int, default=80, help='Sum of flanking sequence lengths on each side of input (i.e. 40+40)')
    parser_variant.add_argument('--model-type', '-t', type=str, choices=['keras', 'pytorch'], default='pytorch', help='Type of model file (keras or pytorch)')
    parser_variant.add_argument('--precision', '-p', type=int, default=2, help='Number of decimal places to round the output scores')

    args = parser_variant.parse_args()

    variant(args)