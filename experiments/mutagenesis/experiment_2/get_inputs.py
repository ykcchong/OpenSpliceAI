import argparse
from pyfaidx import Fasta
import gffutils
import os

donor_motifs = {}
acceptor_motifs = {}    

def get_sequence(seqid, center_pos, half_len, fasta):
    start = center_pos - half_len
    end = center_pos + half_len
    if start < 1:
        return None
    if end > len(fasta[seqid]):
        return None
    seq = fasta[seqid][start:end]
    return seq

def extract_sequences(fasta_file, gff_file, output_donor, output_acceptor, seq_length=400, db_file='experiments/mutagenesis/experiment_2/data/chr1.db'):
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
        
    donor_seqs = []
    acceptor_seqs = []

    # Function to get the biotype of a feature
    def get_biotype(feature):
        for key in ['gene_biotype', 'gene_type', 'biotype']:
            if key in feature.attributes:
                return feature.attributes[key][0]
        return None

    # Iterate over genes in the GFF database
    for gene in db.features_of_type('gene'):
        biotype = get_biotype(gene)
        strand = gene.strand
        gene_seq = fasta[gene.seqid][gene.start-1:gene.end]
        if strand == '-':
            continue
            gene_seq = gene_seq.reverse.complement
        gene_seq = str(gene_seq).upper()
        gene_len = len(gene_seq)
        if biotype == 'protein_coding':
            # Process exons of protein-coding genes
            exons = list(db.children(gene, featuretype='exon', order_by='start'))
            for exon_1, exon_2 in zip(exons[:-1], exons[1:]):
                seqid = exon_1.seqid
                
                first = exon_1.end - gene.start
                second = exon_2.start - gene.start
            
                if strand == '+':
                    donor = first
                    acceptor = second
                elif strand == '-':
                    donor = gene_len - second - 1
                    acceptor = gene_len - first - 1
                
                # Get motifs
                d_motif = str(gene_seq[donor+1:donor+3])
                a_motif = str(gene_seq[acceptor-2:acceptor])  
                if not (d_motif and a_motif):
                    continue   
                donor_motifs[d_motif] = donor_motifs.get(d_motif, 0) + 1
                acceptor_motifs[a_motif] = acceptor_motifs.get(a_motif, 0) + 1
                    
                # Get donor sequence (exon end is the donor site, GT motif)
                donor_seq = get_sequence(seqid, donor + 1, seq_length // 2, fasta)
                acceptor_seq = get_sequence(seqid, acceptor - 1, seq_length // 2, fasta)
                if donor_seq:
                    donor_seqs.append(f">{seqid}_donor_{donor}\n{donor_seq}")
                if acceptor_seq:
                    acceptor_seqs.append(f">{seqid}_acceptor_{acceptor}\n{acceptor_seq}")     
        
    print("Donor motifs:")
    print(sorted(donor_motifs.items(), key=lambda x: x[1], reverse=True))
    print("Acceptor motifs:")
    print(sorted(acceptor_motifs.items(), key=lambda x: x[1], reverse=True))
                
    # Write sequences to output files
    with open(output_donor, "w") as donor_file:
        donor_file.write("\n".join(donor_seqs))
    
    with open(output_acceptor, "w") as acceptor_file:
        acceptor_file.write("\n".join(acceptor_seqs))

if __name__ == '__main__':
    # Uncomment below to use command-line arguments
    # parser = argparse.ArgumentParser(description="Extract donor and acceptor sequences from FASTA and GFF files")
    # parser.add_argument('-fasta', required=True, help='Path to the input FASTA file')
    # parser.add_argument('-gff', required=True, help='Path to the input GFF file')
    # parser.add_argument('-donor', required=True, help='Path to the output donor sequences file')
    # parser.add_argument('-acceptor', required=True, help='Path to the output acceptor sequences file')
    # args = parser.parse_args()
    # extract_sequences(args.fasta, args.gff, args.donor, args.acceptor)
    
    fasta = '/ccb/cybertron/smao10/openspliceai/data/toy/human/chr1.fa'
    gff = '/ccb/cybertron/smao10/openspliceai/data/toy/human/chr1.gff'
    donor = '/ccb/cybertron/smao10/openspliceai/experiments/mutagenesis/experiment_2/data/donor_new.fa'
    acceptor = '/ccb/cybertron/smao10/openspliceai/experiments/mutagenesis/experiment_2/data/acceptor_new.fa'
    extract_sequences(fasta, gff, donor, acceptor)