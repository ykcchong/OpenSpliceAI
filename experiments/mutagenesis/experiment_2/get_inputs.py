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
        start = 1
    if end > len(fasta[seqid]):
        end = len(fasta[seqid])
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

    half_len = seq_length // 2

    # Function to get the biotype of a feature
    def get_biotype(feature):
        for key in ['gene_biotype', 'gene_type', 'biotype']:
            if key in feature.attributes:
                return feature.attributes[key][0]
        return None

    # Iterate over genes in the GFF database
    for gene in db.features_of_type('gene'):
        biotype = get_biotype(gene)
        if biotype == 'protein_coding':
            # Process exons of protein-coding genes
            for exon in db.children(gene, featuretype='exon', order_by='start'):
                seqid = exon.seqid
                start = exon.start
                end = exon.end
                strand = exon.strand

                # Get donor sequence (exon end is the donor site, GT motif)
                donor_seq = get_sequence(seqid, end, half_len, fasta)
                if len(donor_seq) < 2:
                    continue
                if strand == "-":
                    donor_seq = donor_seq.reverse.complement
                donor_seq = str(donor_seq)
                
                if strand == "-":
                    # Check for AG motif (acceptor)
                    mid_acceptor_seq = donor_seq[-2:]  # Get the last two bases for acceptor
                    if mid_acceptor_seq == "AG":
                        acceptor_motifs[mid_acceptor_seq] = acceptor_motifs.get(mid_acceptor_seq, 0) + 1
                        acceptor_seqs.append(f">{seqid}_acceptor_{start}\n{donor_seq}")
                else:
                    # Check for GT motif (donor)
                    mid_donor_seq = donor_seq[-2:]  # Get the last two bases for donor
                    if mid_donor_seq == "GT":
                        donor_motifs[mid_donor_seq] = donor_motifs.get(mid_donor_seq, 0) + 1
                        donor_seqs.append(f">{seqid}_donor_{end}\n{donor_seq}") 

                # Get acceptor sequence (exon start is the acceptor site, AG motif)
                acceptor_seq = get_sequence(seqid, start, half_len, fasta)
                if len(acceptor_seq) < 2:
                    continue
                if strand == "-":
                    acceptor_seq = acceptor_seq.reverse.complement
                acceptor_seq = str(acceptor_seq)

                if strand == "-":
                    # Check for GT motif (donor)
                    mid_donor_seq = acceptor_seq[:2]  # Get the first two bases for donor
                    if mid_donor_seq == "GT":
                        donor_motifs[mid_donor_seq] = donor_motifs.get(mid_donor_seq, 0) + 1
                        donor_seqs.append(f">{seqid}_donor_{end}\n{acceptor_seq}")
                else:
                    # Check for AG motif (acceptor)
                    mid_acceptor_seq = acceptor_seq[:2]  # Get the first two bases for acceptor
                    if mid_acceptor_seq == "AG":
                        acceptor_motifs[mid_acceptor_seq] = acceptor_motifs.get(mid_acceptor_seq, 0) + 1
                        acceptor_seqs.append(f">{seqid}_acceptor_{start}\n{acceptor_seq}")
        
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