"""
create_datafile.py 

- Extracts sequences and labels from GFF-format genome annotations and FASTA-format genomic sequences
- Marks donor and acceptor splice sites, outputs the data in HDF5-format
- Counts the occurrences of motifs at donor and acceptor sites
- Supports handling multiple transcripts per gene, allowing user to choose between processing only longest transcript or all isoforms

Usage:
    Required arguments:
        - annotation_gff: Path to the GFF file containing genome annotations.
        - genome_fasta: Path to the FASTA file containing genome sequences.
        - output_dir: Directory where the output files will be saved.

    Optional arguments:
        - db_file: Specifies the filename for the database. Default is 'gff.db'.
        - parse_type: Choose between processing the 'maximum' transcript length or 'all_isoforms'.

Example:
    python create_datafile.py --annotation_gff path/to/gff --genome_fasta path/to/fasta --output_dir path/to/output

Functions:
    create_or_load_db(gff_file, db_file='gff.db'): Create or load a gffutils database.
    check_and_count_motifs(seq, labels, strand): Check and count motifs at splice sites.
    get_sequences_and_labels(db, fasta_file, output_dir, type, chrom_dict, parse_type="maximum"): Process gene sequences.
    print_motif_counts(): Print the counts of donor and acceptor motifs.
    create_datafile(args): Main function to create the data file from genomic data.
"""

import os 
import gffutils
from Bio import SeqIO
import numpy as np
import h5py
import time
import argparse
from spliceaitoolkit.create_data.utils import *

def create_or_load_db(gff_file, db_file='gff.db'):
    """
    Create a gffutils database from a GFF file, or load it if it already exists.

    Parameters:
    - gff_file: Path to GFF file
    - db_file: Path to save or load the database file (default: 'gff.db')

    Returns:
    - db: gffutils FeatureDB object
    """

    if not os.path.exists(db_file):
        print("Creating new database...")
        db = gffutils.create_db(gff_file, dbfn=db_file, force=True, keep_order=True, merge_strategy='merge', sort_attribute_values=True)
    else:
        print("Loading existing database...")
        db = gffutils.FeatureDB(db_file)
    return db

def get_sequences_and_labels(db, fasta_file, output_dir, type, chrom_dict, parse_type="maximum"):
    """
    Extract sequences for each protein-coding gene, process them based on strand orientation,
    label donor and acceptor sites, and save the data in an HDF5 file.

    Parameters:
    - db: The gffutils database object.
    - fasta_file: Path to the FASTA file.
    - output_dir: Directory to save the output files.
    - type: Type of dataset being processed ('train' or 'test').
    - chrom_dict: Dictionary of chromosomes to process.
    - parse_type: Mode of parsing transcripts ('maximum' or 'all_isoforms').
    """

    seq_dict = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    fw_stats = open(f"{output_dir}stats.txt", "w")
    NAME = []      # Gene Name
    CHROM = []     # Chromosome
    STRAND = []    # Strand in which the gene lies (+ or -)
    TX_START = []  # Position where transcription starts
    TX_END = []    # Position where transcription ends
    SEQ = []       # Nucleotide sequence
    LABEL = []     # Label for each nucleotide in the sequence
    h5f = h5py.File(output_dir + f'datafile_{type}.h5', 'w')
    GENE_COUNTER = 0
    for gene in db.features_of_type('gene'):
        if gene.attributes["gene_biotype"][0] == "protein_coding" and gene.seqid in chrom_dict:
            chrom_dict[gene.seqid] += 1
            gene_id = gene.id
            gene_seq = seq_dict[gene.seqid].seq[gene.start-1:gene.end].upper()  # Extract gene sequence
            labels = [0] * len(gene_seq)  # Initialize all labels to 0
            transcripts = list(db.children(gene, featuretype='mRNA', order_by='start'))
            if len(transcripts) == 0:
                continue
            elif len(transcripts) > 1:
                print(f"Gene {gene_id} has multiple transcripts: {len(transcripts)}")
            ############################################
            # Selecting which mode to process the data
            ############################################
            transcripts_ls = []
            if parse_type == 'maximum':
                max_trans = transcripts[0]
                max_len = max_trans.end - max_trans.start + 1
                for transcript in transcripts:
                    if transcript.end - transcript.start + 1 > max_len:
                        max_trans = transcript
                        max_len = transcript.end - transcript.start + 1
                transcripts_ls = [max_trans]
            elif parse_type == 'all_isoforms':
                transcripts_ls = transcripts
            # Process transcripts
            for transcript in transcripts_ls:
                exons = list(db.children(transcript, featuretype='exon', order_by='start'))
                if len(exons) > 1:
                    GENE_COUNTER += 1
                    for i in range(len(exons) - 1):
                        # Donor site is one base after the end of the current exon
                        first_site = exons[i].end - gene.start  # Adjusted for python indexing
                        # Acceptor site is at the start of the next exon
                        second_site = exons[i + 1].start - gene.start  # Adjusted for python indexing
                        if gene.strand == '+':
                            labels[first_site] = 2  # Mark donor site
                            labels[second_site] = 1  # Mark acceptor site
                        elif gene.strand == '-':
                            d_idx = len(labels) - second_site-1
                            a_idx = len(labels) - first_site-1
                            labels[d_idx] = 2   # Mark donor site
                            labels[a_idx] = 1  # Mark acceptor site
                            seq = gene_seq.reverse_complement()
                            print("D: ", seq[d_idx-3:  d_idx+4])
                            print("A: ", seq[a_idx-6: a_idx+3])
            if gene.strand == '-':
                gene_seq = gene_seq.reverse_complement() # reverse complement the sequence
            gene_seq = str(gene_seq.upper())
            labels_str = ''.join(str(num) for num in labels)
            NAME.append(gene_id)
            CHROM.append(gene.seqid)
            STRAND.append(gene.strand)
            TX_START.append(str(gene.start))
            TX_END.append(str(gene.end))
            SEQ.append(gene_seq)
            LABEL.append(labels_str)
            fw_stats.write(f"{gene.seqid}\t{gene.start}\t{gene.end}\t{gene.id}\t{1}\t{gene.strand}\n")
            check_and_count_motifs(gene_seq, labels, gene.strand)
    fw_stats.close()
    dt = h5py.string_dtype(encoding='utf-8')
    h5f.create_dataset('NAME', data=np.asarray(NAME, dtype=dt) , dtype=dt)
    h5f.create_dataset('CHROM', data=np.asarray(CHROM, dtype=dt) , dtype=dt)
    h5f.create_dataset('STRAND', data=np.asarray(STRAND, dtype=dt) , dtype=dt)
    h5f.create_dataset('TX_START', data=np.asarray(TX_START, dtype=dt) , dtype=dt)
    h5f.create_dataset('TX_END', data=np.asarray(TX_END, dtype=dt) , dtype=dt)
    h5f.create_dataset('SEQ', data=np.asarray(SEQ, dtype=dt) , dtype=dt)
    h5f.create_dataset('LABEL', data=np.asarray(LABEL, dtype=dt) , dtype=dt)
    h5f.close()

def create_datafile(args):
    """
    Main function to create the HDF5 data files for training and testing datasets.
    
    This function takes command-line arguments to specify the input and output directories, 
    as well as the genome annotation and sequence files. It first ensures the output directory exists,
    then creates or loads a gffutils database from the annotation GFF file. It divides the chromosomes
    into training and testing groups, processes gene sequences for each group, labels the donor and
    acceptor splice sites, and finally, writes the processed data to HDF5 files.
    
    Parameters:
    - args (argparse.Namespace): Command-line arguments 
        - output_dir (str): The directory where the HDF5 files will be saved.
        - annotation_gff (str): Path to the GFF file containing genome annotations.
        - genome_fasta (str): Path to the FASTA file containing genome sequences.
        - parse_type (str): Specifies how to handle genes with multiple transcripts ('maximum' or 'all_isoforms').
    """

    ''' 
    NOTE!!!!!!!!!!!!!!!!!!!!!!!!!!
    Where are the `get_all_chromosomes` and `split_chromosomes` functions? nvm they're in the main spliceaitoolkit.py file
    '''
    # Use gffutils to parse annotation file
    os.makedirs(args.output_dir, exist_ok=True)
    db = create_or_load_db(args.annotation_gff, db_file=f'{args.annotation_gff}_db')

    # Find all distinct chromosomes and split them
    all_chromosomes = get_all_chromosomes(db)
    TRAIN_CHROM_GROUP, TEST_CHROM_GROUP = split_chromosomes(all_chromosomes, method='random')  # Or any other method you prefer
    print("TRAIN_CHROM_GROUP: ", TRAIN_CHROM_GROUP)
    print("TEST_CHROM_GROUP: ", TEST_CHROM_GROUP)

    # Collect sequences and labels
    print("--- Step 1: Creating datafile.h5 ... ---")
    start_time = time.time()
    get_sequences_and_labels(db, args.genome_fasta, args.output_dir, type="train", chrom_dict=TRAIN_CHROM_GROUP, parse_type=args.parse_type)
    get_sequences_and_labels(db, args.genome_fasta, args.output_dir, type="test", chrom_dict=TEST_CHROM_GROUP, parse_type=args.parse_type)
    print_motif_counts()
    print("--- %s seconds ---" % (time.time() - start_time))