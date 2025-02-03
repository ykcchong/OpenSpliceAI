import sys
import json
from Bio import SeqIO
from Bio.Seq import Seq
from collections import defaultdict

def parse_gff(gff_file):
    transcripts = defaultdict(list)
    with open(gff_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            seqid = fields[0]
            type_ = fields[2]
            if type_ != 'exon':
                continue
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]
            attributes = fields[8]
            parent = None
            for attr in attributes.split(';'):
                attr = attr.strip()
                if attr.startswith('Parent='):
                    parent = attr.split('=', 1)[1]
                    parents = parent.split(',')
                    for p in parents:
                        transcripts[p].append((seqid, start, end, strand))
                    break
    return transcripts

def process_transcripts(transcripts, genome):
    donors = set()
    acceptors = set()
    
    for tx_id, exons in transcripts.items():
        if not exons:
            continue
        strand = exons[0][3]
        if strand == '+':
            exons_sorted = sorted(exons, key=lambda x: x[1])
        else:
            exons_sorted = sorted(exons, key=lambda x: x[1], reverse=True)
        
        for i in range(len(exons_sorted) - 1):
            current = exons_sorted[i]
            next_ = exons_sorted[i+1]
            current_seqid, current_start, current_end, current_strand = current
            next_seqid, next_start, next_end, next_strand = next_
            
            if current_seqid != next_seqid:
                continue
            
            if strand == '+':
                intron_start = current_end + 1
                intron_end = next_start - 1
                if intron_start > intron_end:
                    continue
                d_start = current_end + 1
                d_end = d_start + 1
                a_start = next_start - 2
                a_end = next_start - 1
            else:
                intron_start = next_end + 1
                intron_end = current_start - 1
                if intron_start > intron_end:
                    continue
                d_start = current_start - 2
                d_end = current_start - 1
                a_start = next_end + 1
                a_end = next_end + 2
            
            current_seq = genome.get(current_seqid)
            if not current_seq:
                continue
            seq_len = len(current_seq.seq)
            
            if d_start < 1 or d_end > seq_len:
                continue
            if a_start < 1 or a_end > seq_len:
                continue
            
            donors.add((current_seqid, d_start, d_end, strand))
            acceptors.add((current_seqid, a_start, a_end, strand))
    
    return donors, acceptors

def count_canonical(sites, genome, site_type):
    canonical = 0
    noncanonical = 0
    motif_counts = defaultdict(int)
    for site in sites:
        seqid, start, end, strand = site
        seq_record = genome.get(seqid)
        if not seq_record:
            continue
        seq = str(seq_record.seq[start-1:end].upper())
        # Reverse complement for reverse strand
        if strand == '-':
            seq = str(Seq(seq).reverse_complement())
        motif_counts[seq] += 1
        # Check canonical status after possible reverse complement
        if site_type == 'donor':
            if seq == 'GT':
                canonical += 1
            else:
                noncanonical += 1
        elif site_type == 'acceptor':
            if seq == 'AG':
                canonical += 1
            else:
                noncanonical += 1
    return canonical, noncanonical, motif_counts

def main(gff_file, fasta_file):
    genome = SeqIO.to_dict(SeqIO.parse(fasta_file, 'fasta'))
    transcripts = parse_gff(gff_file)
    donors, acceptors = process_transcripts(transcripts, genome)
    
    canonical_d, non_d, motifs_d = count_canonical(donors, genome, 'donor')
    canonical_a, non_a, motifs_a = count_canonical(acceptors, genome, 'acceptor')
    
    # Return results for external handling
    return {
        "donor": {
            "canonical": canonical_d,
            "non_canonical": non_d,
            "motifs": dict(motifs_d)
        },
        "acceptor": {
            "canonical": canonical_a,
            "non_canonical": non_a,
            "motifs": dict(motifs_a)
        }
    }

if __name__ == '__main__':
    species = ["Human-MANE", "Mouse", "Zebrafish", "Honeybee", "Thale cress"]
    
    all_donors = {}
    all_acceptors = {}

    for sp in species:
        print(f"Processing {sp}")
        if sp == "Human-MANE":
            genome = "/home/kchao10/data_ssalzbe1/khchao/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna"
            annotation = "/home/kchao10/data_ssalzbe1/khchao/ref_genome/homo_sapiens/MANE/v1.3/MANE.GRCh38.v1.3.refseq_genomic.gff"
        elif sp == "Mouse":
            genome = "/home/kchao10/data_ssalzbe1/khchao/ref_genome/mouse/GCF_000001635.27_GRCm39_genomic.fna"
            annotation = "/home/kchao10/data_ssalzbe1/khchao/ref_genome/mouse/GCF_000001635.27_GRCm39_genomic.gff"
        elif sp == "Zebrafish":
            genome = "/home/kchao10/data_ssalzbe1/khchao/ref_genome/zebra_fish/GCF_000002035.6_GRCz11_genomic.fna"
            annotation = "/home/kchao10/data_ssalzbe1/khchao/ref_genome/zebra_fish/GCF_000002035.6_GRCz11_genomic.gff"
        elif sp == "Honeybee":
            genome = "/home/kchao10/data_ssalzbe1/khchao/ref_genome/bee/HAv3.1_genomic.fna"
            annotation = "/home/kchao10/data_ssalzbe1/khchao/ref_genome/bee/HAv3.1_genomic.gff"
        elif sp == "Thale cress":
            genome = "/home/kchao10/data_ssalzbe1/khchao/ref_genome/arabadop/TAIR10.fna"
            annotation = "/home/kchao10/data_ssalzbe1/khchao/ref_genome/arabadop/TAIR10.gff"

        results = main(annotation, genome)
        
        # Store results
        all_donors[sp] = results["donor"]["motifs"]
        all_acceptors[sp] = results["acceptor"]["motifs"]
        
        # Print summary
        print(f"Donor sites: canonical={results['donor']['canonical']}, non-canonical={results['donor']['non_canonical']}")
        print(f"Acceptor sites: canonical={results['acceptor']['canonical']}, non-canonical={results['acceptor']['non_canonical']}")

    # Save to JSON files
    with open("donor_motifs.json", "w") as f:
        json.dump(all_donors, f, indent=2)
    with open("acceptor_motifs.json", "w") as f:
        json.dump(all_acceptors, f, indent=2)
