import pandas as pd
import random
import matplotlib.pyplot as plt
import logomaker
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
import gffutils
import os

from openspliceai.predict.spliceai import *
from openspliceai.predict.utils import *
from openspliceai.constants import *

from keras.models import load_model
from pkg_resources import resource_filename
from spliceai.utils import one_hot_encode

def read_fasta(fasta_file):
    """Read a FASTA file and return a dictionary of sequences."""
    print(f"Reading FASTA file: {fasta_file}")
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences

def load_gff_db(gff_db_file):
    """Load a pre-built GFF database."""
    print(f"Loading GFF database: {gff_db_file}")
    return gffutils.FeatureDB(gff_db_file)

def extract_gene_sequences_and_splice_sites(gene_ids, sequences, gff_db):
    """Extract gene sequences and splice sites for multiple genes based on GFF annotations."""
    gene_data = {}
    for gene_id in gene_ids:
        gene = gff_db[gene_id]
        chrom = gene.seqid
        gene_start = gene.start - 1  # Convert to 0-based
        gene_end = gene.end
        strand = gene.strand
        
        sequence = sequences[chrom][gene_start:gene_end]
        
        # Extract exon coordinates
        exons = sorted(list(gff_db.children(gene, featuretype='exon')), key=lambda x: x.start)
        
        donor_sites = []
        acceptor_sites = []
        
        if strand == '+':
            for i in range(len(exons) - 1):
                donor_sites.append(exons[i].end - gene_start)
                acceptor_sites.append(exons[i+1].start - gene_start - 1)
        else:
            sequence = str(Seq(sequence).reverse_complement())
            for i in range(len(exons) - 1):
                donor_sites.append(gene_end - exons[i+1].start)
                acceptor_sites.append(gene_end - exons[i].end - 1)
        
        gene_data[gene_id] = {
            'sequence': sequence,
            'donor_sites': donor_sites,
            'acceptor_sites': acceptor_sites
        }
    
    return gene_data

def mutate_sequence(sequence, position):
    """Mutate the sequence at the given position."""
    bases = 'ATCG'
    original_base = sequence[position]
    mutated_bases = [b for b in bases if b != original_base]
    mutated_sequences = []
    for new_base in mutated_bases:
        mutated_seq = sequence[:position] + new_base + sequence[position+1:]
        mutated_sequences.append((new_base, mutated_seq))
    return mutated_sequences

def spliceai_score_batch(sequences, models):
    """Calculate the spliceai scores for multiple sequences."""
    context = 10000
    x = np.stack([one_hot_encode('N'*(context//2) + seq + 'N'*(context//2)) for seq in sequences])
    y = np.mean([models[m].predict(x) for m in range(5)], axis=0)
    acceptor_probs = y[:, :, 1]
    donor_probs = y[:, :, 2]
    return donor_probs, acceptor_probs

def log2_fc(x):
    """Calculate log2 fold change."""
    return np.log2(x / (1-x))

def evaluate_mutations_batch(original_seqs, mutated_seqs, positions, models):
    """Evaluate the effect of mutations using SpliceAI scores for multiple sequences."""
    origin_donor_probs, origin_acceptor_probs = spliceai_score_batch(original_seqs, models)
    mutated_donor_probs, mutated_acceptor_probs = spliceai_score_batch(mutated_seqs, models)
    
    delta_donor_probs = np.array([log2_fc(origin_donor_probs[i, pos]) - log2_fc(mutated_donor_probs[i, pos]) 
                                  for i, pos in enumerate(positions)])
    delta_acceptor_probs = np.array([log2_fc(origin_acceptor_probs[i, pos]) - log2_fc(mutated_acceptor_probs[i, pos]) 
                                     for i, pos in enumerate(positions)])
    
    return delta_donor_probs, delta_acceptor_probs

def in_silico_mutagenesis_batch(gene_data):
    """Perform in silico mutagenesis on multiple sequences."""
    paths = ['models/spliceai{}.h5'.format(x) for x in range(1, 6)]
    models = [load_model(resource_filename('spliceai', x)) for x in paths]
    
    pwm_results = {}
    
    for gene_id, data in gene_data.items():
        sequence = data['sequence']
        pwm_donor = np.zeros((len(sequence), 4))
        pwm_acceptor = np.zeros((len(sequence), 4))
        
        original_seqs = []
        mutated_seqs = []
        positions = []
        
        for i in range(len(sequence)):
            original_base = sequence[i].upper()
            if original_base not in 'ACGT':
                print(f"Skipping non-standard base '{original_base}' at position {i} in gene {gene_id}")
                continue
            
            mutated_sequences = mutate_sequence(sequence, i)
            for new_base, mutated_seq in mutated_sequences:
                original_seqs.append(sequence)
                mutated_seqs.append(mutated_seq)
                positions.append(i)
        
        delta_donor_probs, delta_acceptor_probs = evaluate_mutations_batch(original_seqs, mutated_seqs, positions, models)
        
        idx = 0
        for i in range(len(sequence)):
            original_base = sequence[i].upper()
            if original_base not in 'ACGT':
                continue
            
            for new_base in 'ACGT':
                if new_base != original_base:
                    pwm_donor[i, 'ACGT'.index(new_base)] = delta_donor_probs[idx]
                    pwm_acceptor[i, 'ACGT'.index(new_base)] = delta_acceptor_probs[idx]
                    idx += 1
            
            # Set the score for the original base to 0 (no change)
            pwm_donor[i, 'ACGT'.index(original_base)] = 0
            pwm_acceptor[i, 'ACGT'.index(original_base)] = 0
        
        pwm_results[gene_id] = {
            'pwm_donor': pwm_donor,
            'pwm_acceptor': pwm_acceptor
        }
    
    return pwm_results

def save_pwm(pwm, filename):
    """Save PWM to a file."""
    np.save(filename, pwm)
    print(f"Saved PWM to {filename}")

def plot_dna_logo(gene_id, sequence, pwm_donor, pwm_acceptor, donor_sites, acceptor_sites):
    """Plot DNA logo with mutation effects and labeled splice sites."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(200, 10))
    
    for idx, (pwm, title) in enumerate([(pwm_donor, 'Donor'), (pwm_acceptor, 'Acceptor')]):
        # Convert numpy array to pandas DataFrame
        df = pd.DataFrame(pwm, columns=['A', 'C', 'G', 'T'])
        
        # Create logo
        logo = logomaker.Logo(df, ax=ax1 if idx == 0 else ax2)
        logo.style_spines(visible=False)
        logo.style_xticks(rotation=90, fmt='%d', anchor=0)
        
        ax = ax1 if idx == 0 else ax2
        ax.set_ylabel("Delta score")
        ax.set_title(f"{title} site logo")
        
        # Add splice site annotations
        ymin, ymax = ax.get_ylim()
        for site in donor_sites:
            ax.axvline(x=site, color='red', linestyle='--', alpha=0.5)
            ax.text(site, ymax, 'D', color='red', ha='center', va='bottom')
        for site in acceptor_sites:
            ax.axvline(x=site, color='blue', linestyle='--', alpha=0.5)
            ax.text(site, ymax, 'A', color='blue', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{gene_id}_dna_logo.png")

# Main execution
fasta_file = "/home/kchao10/data_ssalzbe1/khchao/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna"
gff_file = "/home/kchao10/data_ssalzbe1/khchao/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.gff_db"
gene_ids = ["gene-LINC01409", "gene-BRCA1", "gene-TP53"]  # Add more gene IDs as needed

# Read input files
sequences = read_fasta(fasta_file)
gff_db = load_gff_db(gff_file)

# Extract gene sequences and splice sites for multiple genes
gene_data = extract_gene_sequences_and_splice_sites(gene_ids, sequences, gff_db)

# Perform in silico mutagenesis for multiple genes
pwm_results = in_silico_mutagenesis_batch(gene_data)

# Save PWMs and plot DNA logos for each gene
for gene_id, pwm_data in pwm_results.items():
    save_pwm(pwm_data['pwm_donor'], os.path.join(f"{gene_id}_pwm_donor.npy"))
    save_pwm(pwm_data['pwm_acceptor'], os.path.join(f"{gene_id}_pwm_acceptor.npy"))
    
    plot_dna_logo(gene_id, 
                  gene_data[gene_id]['sequence'], 
                  pwm_data['pwm_donor'], 
                  pwm_data['pwm_acceptor'], 
                  gene_data[gene_id]['donor_sites'], 
                  gene_data[gene_id]['acceptor_sites'])

    # Print PWM matrices
    print(f"\nPWM matrices for {gene_id}:")
    print("Donor PWM:")
    print(pwm_data['pwm_donor'])
    print("\nAcceptor PWM:")
    print(pwm_data['pwm_acceptor'])
    break
# import pandas as pd
# import random
# import matplotlib.pyplot as plt
# import logomaker
# import numpy as np
# from Bio import SeqIO
# from Bio.Seq import Seq
# import gffutils
# import os

# from openspliceai.predict.spliceai import *
# from openspliceai.predict.utils import *
# from openspliceai.constants import *

# from keras.models import load_model
# from pkg_resources import resource_filename
# from spliceai.utils import one_hot_encode

# def read_fasta(fasta_file):
#     """Read a FASTA file and return a dictionary of sequences."""
#     sequences = {}
#     for record in SeqIO.parse(fasta_file, "fasta"):
#         sequences[record.id] = str(record.seq)
#     return sequences

# def load_gff_db(gff_db_file):
#     """Load a pre-built GFF database."""
#     return gffutils.FeatureDB(gff_db_file)

# def extract_gene_sequence_and_splice_sites(gene_id, sequences, gff_db):
#     """Extract the gene sequence and splice sites based on GFF annotations."""
#     gene = gff_db[gene_id]
#     chrom = gene.seqid
#     gene_start = gene.start - 1  # Convert to 0-based
#     gene_end = gene.end
#     strand = gene.strand
    
#     sequence = sequences[chrom][gene_start:gene_end]
    
#     # Extract exon coordinates
#     exons = sorted(list(gff_db.children(gene, featuretype='exon')), key=lambda x: x.start)
    
#     donor_sites = []
#     acceptor_sites = []
    
#     if strand == '+':
#         for i in range(len(exons) - 1):
#             donor_sites.append(exons[i].end - gene_start)
#             acceptor_sites.append(exons[i+1].start - gene_start - 1)
#     else:
#         sequence = str(Seq(sequence).reverse_complement())
#         for i in range(len(exons) - 1):
#             donor_sites.append(gene_end - exons[i+1].start)
#             acceptor_sites.append(gene_end - exons[i].end - 1)
    
#     return sequence, donor_sites, acceptor_sites


# def generate_sequence(length=4):
#     """Generate a random DNA sequence with a donor site in the middle."""
#     donor_site = "AGGT"
#     half_length = (length - len(donor_site)) // 2
#     upstream = ''.join(random.choice('ATCG') for _ in range(half_length))
#     downstream = ''.join(random.choice('ATCG') for _ in range(half_length))
#     return upstream + donor_site + downstream

# def mutate_sequence(sequence, position):
#     """Mutate the sequence at the given position."""
#     bases = 'ATCG'
#     original_base = sequence[position]
#     mutated_bases = [b for b in bases if b != original_base]
#     mutated_sequences = []
#     for new_base in mutated_bases:
#         mutated_seq = sequence[:position] + new_base + sequence[position+1:]
#         mutated_sequences.append((new_base, mutated_seq))
#     return mutated_sequences

# def spliceai_score(sequence, models):
#     """Calculate the spliceai score."""
#     context = 10000
#     x = one_hot_encode('N'*(context//2) + sequence + 'N'*(context//2))[None, :]
#     y = np.mean([models[m].predict(x) for m in range(5)], axis=0)
#     acceptor_prob = y[0, :, 1]
#     donor_prob = y[0, :, 2]
#     return donor_prob, acceptor_prob

# def log2_fc(x):
#     """Calculate log2 fold change."""
#     return np.log2(x / (1-x))

# def evaluate_mutation(original_seq, mutated_seq, position, models):
#     """Evaluate the effect of mutation using SpliceAI scores."""
#     origin_donor_prob, origin_acceptor_prob = spliceai_score(original_seq, models)
#     mutated_donor_prob, mutated_acceptor_prob = spliceai_score(mutated_seq, models)
    
#     # Calculate delta scores at the mutated position
#     delta_donor_prob = log2_fc(origin_donor_prob[position]) - log2_fc(mutated_donor_prob[position])
#     delta_acceptor_prob = log2_fc(origin_acceptor_prob[position]) - log2_fc(mutated_acceptor_prob[position])
#     return delta_donor_prob, delta_acceptor_prob

# def in_silico_mutagenesis(sequence):
#     """Perform in silico mutagenesis on the sequence."""
#     paths = ['models/spliceai{}.h5'.format(x) for x in range(1, 6)]
#     models = [load_model(resource_filename('spliceai', x)) for x in paths]
    
#     pwm_donor = np.zeros((len(sequence), 4))
#     pwm_acceptor = np.zeros((len(sequence), 4))
    
#  for i in range(len(sequence)):
#         original_base = sequence[i].upper()
#         if original_base not in 'ACGT':
#             print(f"Skipping non-standard base '{original_base}' at position {i}")
#             continue
        
#         mutated_sequences = mutate_sequence(sequence, i)
#         for new_base, mutated_seq in mutated_sequences:
#             delta_donor, delta_acceptor = evaluate_mutation(sequence, mutated_seq, i, models)
#             pwm_donor[i, 'ACGT'.index(new_base)] = delta_donor
#             pwm_acceptor[i, 'ACGT'.index(new_base)] = delta_acceptor
        
#         # Set the score for the original base to 0 (no change)
#         pwm_donor[i, 'ACGT'.index(original_base)] = 0
#         pwm_acceptor[i, 'ACGT'.index(original_base)] = 0
    
#     return pwm_donor, pwm_acceptor

# def save_pwm(pwm, filename):
#     """Save PWM to a file."""
#     np.save(filename, pwm)
#     print(f"Saved PWM to {filename}")

# def plot_dna_logo(gene_id, sequence, pwm_donor, pwm_acceptor, donor_sites, acceptor_sites):
#     """Plot DNA logo with mutation effects and labeled splice sites."""
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(200, 10))
    
#     for idx, (pwm, title) in enumerate([(pwm_donor, 'Donor'), (pwm_acceptor, 'Acceptor')]):
#         # Convert numpy array to pandas DataFrame
#         df = pd.DataFrame(pwm, columns=['A', 'C', 'G', 'T'])
        
#         # Create logo
#         logo = logomaker.Logo(df, ax=ax1 if idx == 0 else ax2)
#         logo.style_spines(visible=False)
#         logo.style_xticks(rotation=90, fmt='%d', anchor=0)
        
#         ax = ax1 if idx == 0 else ax2
#         ax.set_ylabel("Delta score")
#         ax.set_title(f"{title} site logo")
        
#         # Add splice site annotations
#         ymin, ymax = ax.get_ylim()
#         for site in donor_sites:
#             ax.axvline(x=site, color='red', linestyle='--', alpha=0.5)
#             ax.text(site, ymax, 'D', color='red', ha='center', va='bottom')
#         for site in acceptor_sites:
#             ax.axvline(x=site, color='blue', linestyle='--', alpha=0.5)
#             ax.text(site, ymax, 'A', color='blue', ha='center', va='bottom')
    
#     plt.tight_layout()
#     plt.savefig(f"{gene_id}_dna_logo.png")

# # Main execution
# fasta_file = "/home/kchao10/data_ssalzbe1/khchao/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna"
# gff_file = "/home/kchao10/data_ssalzbe1/khchao/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.gff_db"
# gene_id = "gene-LINC01409"

# # Read input files
# sequences = read_fasta(fasta_file)
# gff_db = load_gff_db(gff_file)

# # Extract gene sequence and splice sites
# gene_sequence, donor_sites, acceptor_sites = extract_gene_sequence_and_splice_sites(gene_id, sequences, gff_db)
# print(f"Extracted gene sequence (length: {len(gene_sequence)}):")
# print(gene_sequence[:50] + "..." + gene_sequence[-50:])
# print(f"Donor sites: {donor_sites}")
# print(f"Acceptor sites: {acceptor_sites}")


# # Perform in silico mutagenesis
# pwm_donor, pwm_acceptor = in_silico_mutagenesis(gene_sequence)

# save_pwm(pwm_donor, os.path.join(f"{gene_id}_pwm_donor.npy"))
# save_pwm(pwm_acceptor, os.path.join(f"{gene_id}_pwm_acceptor.npy"))

# # Plot DNA logo with labeled splice sites
# plot_dna_logo(gene_id, gene_sequence, pwm_donor, pwm_acceptor, donor_sites, acceptor_sites)

# # Print PWM matrices
# print("Donor PWM:")
# print(pwm_donor)
# print("\nAcceptor PWM:")
# print(pwm_acceptor)
