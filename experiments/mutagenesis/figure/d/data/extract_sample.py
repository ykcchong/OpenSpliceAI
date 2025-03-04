from pyfaidx import Fasta

def extract_region(fasta_path, chromosome, start, end, reverse):
    # Load the fasta file
    fasta = Fasta(fasta_path, sequence_always_upper=True, rebuild=False)
    # Extract the specified region (1-based indexing in pyfaidx)
    if reverse:
        sequence = fasta[chromosome][start-1:end].reverse.complement.seq
    else:
        sequence = fasta[chromosome][start-1:end].seq
    return sequence
    

# MYBPC3 gene (GRCh38)
# name = 'mybpc3'
# chromosome = "chr11" 
# start = 47331405
# end = 47352702 

# CFTR gene (GRCh38)
name = 'cftr'
chromosome = 'chr7'
start = 117480024
end = 117668665

fasta_path = "/ccb/cybertron2/smao10/openspliceai/data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna"
print('len', end-start)
reverse = False
reverse_char = '(-)' if reverse else '(+)'

window = 0
flanking = 10000
extract_start = start - (window // 2) - (flanking // 2) 
extract_end = end + (window // 2) + (flanking // 2) - 1

# Extract sequence and print
sequence = extract_region(fasta_path, chromosome, extract_start, extract_end, reverse)
if sequence:
    output_file = f"/ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/e/data/{name}.fa"

    with open(output_file, "w") as file:
        file.write(f'>{chromosome}:{extract_start}-{extract_end}{reverse_char}_{name}_{window}window_{flanking}flank\n')
        file.write(sequence)