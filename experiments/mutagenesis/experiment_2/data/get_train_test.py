from pyfaidx import Fasta
import gffutils
import os

donor_motifs = {}
acceptor_motifs = {}    

def reverse_complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq))

def get_sequence(gene_seq, center_pos, half_len):

    # Assume all is forward strand 
    
    start = center_pos - half_len
    end = center_pos + half_len
    if start < 0:
        return None
    if end > len(gene_seq):
        return None
    seq = gene_seq[start:end]
    return seq

def extract_sequences(fasta_file, gff_file, output_donor_train, output_acceptor_train, output_donor_test, output_acceptor_test, db_file, seq_length=400):
    # Load the FASTA file
    fasta = Fasta(fasta_file, sequence_always_upper=True)
    
    # Create a gffutils database from the GFF file
    if os.path.exists(db_file):
        print(f"Connecting to existing GFF database: {db_file}")
        db = gffutils.FeatureDB(db_file)
    else:
        print(f"Creating new GFF database: {db_file}")
        db = gffutils.create_db(
            gff_file,
            dbfn=db_file,
            force=True,
            keep_order=True,
            merge_strategy='create_unique',
            sort_attribute_values=True,
            verbose=True
        )
        
    donor_seqs_train = []
    acceptor_seqs_train = []
    donor_seqs_test = []
    acceptor_seqs_test = []

    # Function to get the biotype of a feature
    def get_biotype(feature):
        for key in ['gene_biotype', 'gene_type', 'biotype']:
            if key in feature.attributes:
                return feature.attributes[key][0]
        return None

    # Determine training and testing chromosomes
    test_chromosomes = {'chr1', 'chr3', 'chr5', 'chr7', 'chr9'}

    # Iterate over genes in the GFF database
    skipped = 0
    for gene in db.features_of_type('gene'):
        biotype = get_biotype(gene)
        strand = gene.strand
        gene_seq = fasta[gene.seqid][gene.start - 1:gene.end] # 0-based indexing
        if strand == '-':
            gene_seq = gene_seq.reverse.complement
        gene_seq = str(gene_seq).upper()
        gene_len = len(gene_seq)

        if biotype == 'protein_coding':
            # Process exons of protein-coding genes
            exons = list(db.children(gene, featuretype='exon', order_by='start'))
            for exon_1, exon_2 in zip(exons[:-1], exons[1:]):
                seqid = exon_1.seqid
                is_train = seqid not in test_chromosomes

                # Adjust positions to be relative to gene sequence (0-based indexing)
                exon1_end_rel = exon_1.end - gene.start
                exon2_start_rel = exon_2.start - gene.start

                # Fix positions by strand, so motif always starts in the middle (199-200)
                if strand == '+':
                    donor_pos = exon1_end_rel + 1
                    acceptor_pos = exon2_start_rel - 2
                elif strand == '-':
                    donor_pos = gene_len - exon2_start_rel
                    acceptor_pos = gene_len - exon1_end_rel - 3
                else:
                    print('err: no strand')
                    continue  # Skip if strand is not '+' or '-'

                # Extract donor and acceptor motifs
                d_motif = gene_seq[donor_pos:donor_pos+2]
                a_motif = gene_seq[acceptor_pos:acceptor_pos+2]
                donor_motifs[d_motif] = donor_motifs.get(d_motif, 0) + 1
                acceptor_motifs[a_motif] = acceptor_motifs.get(a_motif, 0) + 1

                # Get donor and acceptor sequences
                donor_seq = get_sequence(gene_seq, donor_pos+1, seq_length // 2)
                acceptor_seq = get_sequence(gene_seq, acceptor_pos+1, seq_length // 2)

                # Only proceed if full sequences are available
                if donor_seq:
                    found_motif = donor_seq[seq_length // 2 - 1: seq_length // 2 + 1]
                    #print(f"Donor motif in sequence: {found_motif}, Expected donor motif: {d_motif}")
                    assert found_motif == d_motif
                    if is_train:
                        donor_seqs_train.append(f">{seqid}_donor_{donor_pos}\n{donor_seq}")
                    else:
                        donor_seqs_test.append(f">{seqid}_donor_{donor_pos}\n{donor_seq}")

                if acceptor_seq:
                    found_motif = acceptor_seq[seq_length // 2 - 1: seq_length // 2 + 1]
                    #print(f"Acceptor motif in sequence: {found_motif}, Expected acceptor motif: {a_motif}")
                    assert found_motif == a_motif
                    if is_train:
                        acceptor_seqs_train.append(f">{seqid}_acceptor_{acceptor_pos}\n{acceptor_seq}")
                    else:
                        acceptor_seqs_test.append(f">{seqid}_acceptor_{acceptor_pos}\n{acceptor_seq}")

    print("Donor motifs:")
    print(sorted(donor_motifs.items(), key=lambda x: x[1], reverse=True))
    print("Acceptor motifs:")
    print(sorted(acceptor_motifs.items(), key=lambda x: x[1], reverse=True))
    print(f"Skipped {skipped} sequences")
                
    # Write sequences to train and test output files
    with open(output_donor_train, "w") as donor_file_train:
        donor_file_train.write("\n".join(donor_seqs_train))
    
    with open(output_acceptor_train, "w") as acceptor_file_train:
        acceptor_file_train.write("\n".join(acceptor_seqs_train))
        
    with open(output_donor_test, "w") as donor_file_test:
        donor_file_test.write("\n".join(donor_seqs_test))
    
    with open(output_acceptor_test, "w") as acceptor_file_test:
        acceptor_file_test.write("\n".join(acceptor_seqs_test))

# File paths
fasta = '/ccb/cybertron/smao10/openspliceai/data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna'
gff = '/ccb/cybertron/smao10/openspliceai/data/ref_genome/homo_sapiens/MANE/v1.3/MANE.GRCh38.v1.3.refseq_genomic.gff'

output_base = '/ccb/cybertron/smao10/openspliceai/experiments/mutagenesis/experiment_2/data'
output_donor_train = f'{output_base}/train_donor.fa'
output_acceptor_train = f'{output_base}/train_acceptor.fa'
output_donor_test = f'{output_base}/test_donor.fa'
output_acceptor_test = f'{output_base}/test_acceptor.fa'
db = f'{output_base}/MANE.db'

# Execute extraction
seq_length = 400 + 10000
extract_sequences(fasta, gff, output_donor_train, output_acceptor_train, output_donor_test, output_acceptor_test, db, seq_length)