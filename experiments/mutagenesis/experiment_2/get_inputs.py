import argparse
from pyfaidx import Fasta
import pandas as pd
import gffutils 

donor_motifs = {}
acceptor_motifs = {}    

def extract_sequences(fasta_file, gff_file, output_donor, output_acceptor, seq_length=400):
    # Load the FASTA and GFF files
    fasta = Fasta(fasta_file, sequence_always_upper=True)
    gff = pd.read_csv(gff_file, sep="\t", header=None, comment="#", 
                      names=["seqid", "source", "feature", "start", "end", "score", "strand", "phase", "attributes"])

    donor_seqs = []
    acceptor_seqs = []

    half_len = seq_length // 2

    # Process each entry in the GFF file
    for _, row in gff.iterrows():
        if row["feature"] == "exon":
            seqid = row["seqid"]
            start, end = row["start"], row["end"]
            strand = row["strand"]

            # Get donor sequence (exon end is the donor site)
            if end + half_len < len(fasta[seqid]):
                donor_seq = fasta[seqid][end - half_len + 1:end + half_len + 1]
                if strand == "-":
                    donor_seq = donor_seq.reverse.complement
                donor_seq = str(donor_seq)
                
                if strand == "-":
                    # Count motifs
                    mid_acceptor_seq = donor_seq[half_len-1:half_len+1]
                    if mid_acceptor_seq in acceptor_motifs:
                        acceptor_motifs[mid_acceptor_seq] += 1
                    else:
                        acceptor_motifs[mid_acceptor_seq] = 1
                    
                    if mid_acceptor_seq == "AG":
                        acceptor_seqs.append(f">{seqid}_acceptor_{start}\n{donor_seq}")
                else:
                    # Count motifs
                    mid_donor_seq = donor_seq[half_len-1:half_len+1]
                    if mid_donor_seq in donor_motifs:
                        donor_motifs[mid_donor_seq] += 1
                    else:
                        donor_motifs[mid_donor_seq] = 1
                    
                    if mid_donor_seq == "GT":
                        donor_seqs.append(f">{seqid}_donor_{end}\n{donor_seq}")   


            # Get acceptor sequence (exon start is the acceptor site)
            if start - half_len > 0:
                acceptor_seq = fasta[seqid][start - half_len:start + half_len]
                if strand == "-":
                    acceptor_seq = acceptor_seq.reverse.complement
                acceptor_seq = str(acceptor_seq)

                if strand == "-":
                    # Count motifs
                    mid_donor_seq = acceptor_seq[half_len-1:half_len+1]
                    if mid_donor_seq in donor_motifs:
                        donor_motifs[mid_donor_seq] += 1
                    else:
                        donor_motifs[mid_donor_seq] = 1
                        
                    if mid_donor_seq == "GT":
                        donor_seqs.append(f">{seqid}_donor_{end}\n{acceptor_seq}")     
                else:  
                    # Count motifs
                    mid_acceptor_seq = acceptor_seq[half_len-1:half_len+1]
                    if mid_acceptor_seq in acceptor_motifs:
                        acceptor_motifs[mid_acceptor_seq] += 1
                    else:
                        acceptor_motifs[mid_acceptor_seq] = 1
                    
                    if mid_acceptor_seq == "AG":
                        acceptor_seqs.append(f">{seqid}_acceptor_{start}\n{acceptor_seq}")
    
    print("Donor motifs:")
    print(sorted(donor_motifs.items(), key=lambda x: x[1], reverse=True))
    print("Acceptor motifs:")
    print(sorted(acceptor_motifs.items(), key=lambda x: x[1], reverse=True))
            
    # Write to output files
    with open(output_donor, "w") as donor_file:
        donor_file.write("\n".join(donor_seqs))
    
    with open(output_acceptor, "w") as acceptor_file:
        acceptor_file.write("\n".join(acceptor_seqs))

def reverse_complement(seq):
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(complement[base] for base in reversed(seq))

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Extract donor and acceptor sequences from FASTA and GFF files")
    # parser.add_argument('-fasta', required=True, help='Path to the input FASTA file')
    # parser.add_argument('-gff', required=True, help='Path to the input GFF file')
    # parser.add_argument('-donor', required=True, help='Path to the output donor sequences file')
    # parser.add_argument('-acceptor', required=True, help='Path to the output acceptor sequences file')
    
    # args = parser.parse_args()
    
    # extract_sequences(args.fasta, args.gff, args.donor, args.acceptor)
    
    fasta = '/ccb/cybertron/smao10/openspliceai/data/toy/human/chr1.fa'
    gff = '/ccb/cybertron/smao10/openspliceai/data/toy/human/chr1.gff'
    donor = '/ccb/cybertron/smao10/openspliceai/experiments/mutagenesis/experiment_2/data/donor.fa'
    acceptor = '/ccb/cybertron/smao10/openspliceai/experiments/mutagenesis/experiment_2/data/acceptor.fa'
    extract_sequences(fasta, gff, donor, acceptor)