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
        - parse_type: Choose between processing the 'canonical' transcript length or 'all_isoforms'.

Example:
    python create_datafile.py --annotation_gff path/to/gff --genome_fasta path/to/fasta --output_dir path/to/output

Functions:
    create_or_load_db(gff_file, db_file='gff.db'): Create or load a gffutils database.
    check_and_count_motifs(seq, labels, strand): Check and count motifs at splice sites.
    get_sequences_and_labels(db, fasta_file, output_dir, type, chrom_dict, parse_type="canonical"): Process gene sequences.
    print_motif_counts(): Print the counts of donor and acceptor motifs.
    create_datafile(args): Main function to create the data file from genomic data.
"""

import os 
from Bio import SeqIO
import numpy as np
import h5py
import time
import argparse
import gffutils
import openspliceai.create_data.utils as utils

donor_motif_counts = {}  # Initialize counts
acceptor_motif_counts = {}  # Initialize counts

def get_sequences_and_labels(db, output_dir, seq_dict, data_type, chrom_dict, parse_type="canonical", biotype="protein-coding", canonical_only=True):
    """
    Extract sequences for each protein-coding gene, reverse complement sequences for genes on the reverse strand,
    and label donor and acceptor sites correctly based on strand orientation.
    """
    fw_stats = open(f"{output_dir}stats.txt", "w")
    NAME = []      # Gene Name
    CHROM = []     # Chromosome
    STRAND = []    # Strand in which the gene lies (+ or -)
    TX_START = []  # Position where transcription starts
    TX_END = []    # Position where transcription ends
    SEQ = []       # Nucleotide sequence
    LABEL = []     # Label for each nucleotide in the sequence
    h5fname = None
    if biotype =="non-coding":
        h5fname = output_dir + f'datafile_{data_type}_ncRNA.h5'
    elif biotype =="protein-coding":
        h5fname = output_dir + f'datafile_{data_type}.h5'
    h5f = h5py.File(h5fname, 'w')
    GENE_COUNTER = 0
    for gene in db.features_of_type('gene'):
        if "exception" in gene.attributes.keys() and gene.attributes["exception"][0] == "trans-splicing":
            continue
        if gene.seqid not in chrom_dict:
            continue
        if biotype =="protein-coding":
            if gene.attributes["gene_biotype"][0] != "protein_coding":
                continue
        elif biotype =="non-coding":
            if gene.attributes["gene_biotype"][0] != "lncRNA" and gene.attributes["gene_biotype"][0] != "ncRNA":
                continue
        else:
            continue
        chrom_dict[gene.seqid] += 1
        gene_id = gene.id

        ############################################
        # Process the gene sequence
        ############################################
        gene_seq = seq_dict[gene.seqid].seq[gene.start-1:gene.end].upper()  # Extract gene sequence
        if gene.strand == '-':
            gene_seq = gene_seq.reverse_complement() # reverse complement the sequence
        gene_seq = str(gene_seq.upper())
        print(f"\tProcessing gene {gene_id} on chromosome {gene.seqid}..., len(gene_seq): {len(gene_seq)}")
        labels = [0] * len(gene_seq)  # Initialize all labels to 0
        if biotype =="protein-coding":
            transcripts = list(db.children(gene, featuretype='mRNA', order_by='start'))
        elif biotype =="non-coding":
            transcripts = list(db.children(gene, level=1, order_by='start'))
        if len(transcripts) == 0:
            continue
        elif len(transcripts) > 1:
            print(f"Gene {gene_id} has multiple transcripts: {len(transcripts)}")
        ############################################
        # Selecting which mode to process the data
        ############################################
        transcripts_ls = []
        if parse_type == 'canonical':
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
                    second_site = exons[i + 1].start - gene.start  
        
                    # Adjusted for python indexing
                    if gene.strand == '+':
                        d_idx = first_site
                        a_idx = second_site
                    elif gene.strand == '-':
                        d_idx = len(labels) - second_site-1
                        a_idx = len(labels) - first_site-1
                        # print(f"Gene {gene_id} is on the reverse strand, d_idx: {d_idx}, a_idx: {a_idx}; len(labels): {len(labels)}")
                        
                    ############################################
                    # Check if the donor / acceptor sites are valid
                    ############################################
                    d_motif = str(gene_seq[d_idx+1:d_idx+3])
                    a_motif = str(gene_seq[a_idx-2:a_idx])
                    if not canonical_only:
                        labels[d_idx] = 2   # Mark donor site
                        labels[a_idx] = 1  # Mark acceptor site
                    elif (d_motif == "GT" and a_motif == "AG") or (d_motif == "GC" and a_motif == "AG") or (d_motif == "AT" and a_motif == "AC"):
                        labels[d_idx] = 2   # Mark donor site
                        labels[a_idx] = 1  # Mark acceptor site
            labels_str = ''.join(str(num) for num in labels)
            NAME.append(gene_id)
            CHROM.append(gene.seqid)
            STRAND.append(gene.strand)
            TX_START.append(str(gene.start))
            TX_END.append(str(gene.end))
            SEQ.append(gene_seq)
            LABEL.append(labels_str)
            fw_stats.write(f"{gene.seqid}\t{gene.start}\t{gene.end}\t{gene.id}\t{1}\t{gene.strand}\n")
            utils.check_and_count_motifs(gene_seq, labels, donor_motif_counts, acceptor_motif_counts)
    print(f"Total SEQ: {len(SEQ)}")
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
        - parse_type (str): Specifies how to handle genes with multiple transcripts ('canonical' or 'all_isoforms').
        - split_method (str): Method for splitting chromosomes into training and testing groups ('random' or 'human').
        - split_ratio (float): Ratio of training and testing datasets.
    """
    print("--- Step 1: Creating datafile.h5 ... ---")
    start_time = time.time()
    
    # Use gffutils to parse annotation file
    os.makedirs(args.output_dir, exist_ok=True)
    db = utils.create_or_load_db(args.annotation_gff, db_file=f'{args.annotation_gff}_db')
    seq_dict = SeqIO.to_dict(SeqIO.parse(args.genome_fasta, "fasta"))
    
    # Find all distinct chromosomes and split them
    all_chromosomes = utils.get_all_chromosomes(db)
    TRAIN_CHROM_GROUP, TEST_CHROM_GROUP = utils.split_chromosomes(all_chromosomes, method=args.split_method, split_ratio=args.split_ratio)
    print("TRAIN_CHROM_GROUP: ", TRAIN_CHROM_GROUP)
    print("TEST_CHROM_GROUP: ", TEST_CHROM_GROUP)

    # Collect sequences and labels for testing and/or training groups
    if args.chr_split == 'test':
        print("Creating test datafile...")
        get_sequences_and_labels(db, args.output_dir, seq_dict, data_type="test", chrom_dict=TEST_CHROM_GROUP, parse_type=args.parse_type, biotype=args.biotype, canonical_only=args.canonical_only)
    elif args.chr_split == 'train-test':
        print("Creating train datafile...")
        get_sequences_and_labels(db, args.output_dir, seq_dict, data_type="train", chrom_dict=TRAIN_CHROM_GROUP, parse_type=args.parse_type, biotype=args.biotype, canonical_only=args.canonical_only)
        print("Creating test datafile...")
        get_sequences_and_labels(db, args.output_dir, seq_dict, data_type="test", chrom_dict=TEST_CHROM_GROUP, parse_type=args.parse_type, biotype=args.biotype, canonical_only=args.canonical_only)
    utils.print_motif_counts(donor_motif_counts, acceptor_motif_counts)
    
    print("--- %s seconds ---" % (time.time() - start_time))