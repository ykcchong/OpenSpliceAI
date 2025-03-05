import random
from pyfaidx import Fasta
import gffutils
import os
from Bio.Seq import Seq

def get_gene(gff_path, db_path, fasta_path, output_base, sample_num, site, window_size=800):
    # Read the input fasta file
    fasta = Fasta(fasta_path, duplicate_action='first', sequence_always_upper=True)
    
    # Connect to the db, or generate it if it does not exist
    if not os.path.exists(db_path):
        print("Creating a database from the GFF file.")
        db = gffutils.create_db(
            gff_path,
            dbfn=db_path,
            force=True,
            keep_order=True,
            merge_strategy='create_unique',
            sort_attribute_values=True,
            verbose=True
        )
    else:
        print("Connecting to the existing database.")
        db = gffutils.FeatureDB(db_path)
    
    # Set the random seed based on sample_num for reproducibility
    random.seed(sample_num)
    
    # Extract all protein-coding gene sequences
    protein_coding_genes = [
        gene for gene in db.features_of_type('gene')
        if 'gene_biotype' in gene.attributes and 'protein_coding' in gene.attributes['gene_biotype']
    ]
    
    if not protein_coding_genes:
        print("No protein-coding genes found in the database.")
        return
    
    # Randomly select a protein-coding gene
    selected_gene = random.choice(protein_coding_genes)
    print(f"Selected gene: {selected_gene.id}")
    
    # Locate the positions of all donor and acceptor sites of the gene
    exons = list(db.children(selected_gene, featuretype='exon', order_by='start'))
    if len(exons) < 2:
        print("Not enough exons to define splice sites.")
        return
    
    strand = selected_gene.strand
    if strand == '+':
        acceptor_sites = [exon.start for exon in exons[1:]]  # Skip the first exon
        donor_sites = [exon.end for exon in exons[:-1]]      # Skip the last exon
    else:
        acceptor_sites = [exon.end for exon in exons[:-1]]   # Skip the last exon
        donor_sites = [exon.start for exon in exons[1:]]     # Skip the first exon
    
    print(f"Acceptor sites: {acceptor_sites}")
    print(f"Donor sites: {donor_sites}")
    
    # Select the site type
    if site == 'donor':
        sites_list = donor_sites
    elif site == 'acceptor':
        sites_list = acceptor_sites
    else:
        raise ValueError("Invalid site type. Should be 'donor' or 'acceptor'.")
    
    if not sites_list:
        print(f"No {site} sites found for the selected gene.")
        return
    
    # Randomly select a site (reproducible due to seed)
    selected_site = random.choice(sites_list)
    print(f"Selected {site} site at position: {selected_site}")
    
    half_window = window_size // 2
    chrom = selected_gene.chrom
    
    chrom_length = len(fasta[chrom])
    start = max(1, selected_site - half_window)
    end = min(selected_site + half_window - 1, chrom_length)
    
    sequence = fasta[chrom][start:end].seq
    
    if strand == '-':
        sequence = str(Seq(sequence).reverse_complement())
    
    # Generate an output filepath based on the site and sample number
    output_file = os.path.join(output_base, f'{site}_{sample_num}.fa')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    header = f'>{chrom}:{start}-{end}({strand})_{site}_site_{selected_site}'
    
    with open(output_file, 'w') as f:
        f.write(f'{header}\n{sequence}\n')
        

########## CHANGE ##########
sample_num = 2
site = 'acceptor'
base = '/ccb/cybertron/smao10/openspliceai'
gff_path = f'{base}/data/ref_genome/homo_sapiens/MANE/v1.3/MANE.GRCh38.v1.3.refseq_genomic.gff'
db_path = f'{base}/data/ref_genome/homo_sapiens/MANE/v1.3/MANE.GRCh38.v1.3.refseq_genomic.gff_db'
fasta_path = f'{base}/data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna'
output_base = f'{base}/experiments/mutagenesis/experiment_1/data'

get_gene(gff_path, db_path, fasta_path, output_base, sample_num, site)
    