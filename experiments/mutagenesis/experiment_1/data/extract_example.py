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
    

# Example usage
fasta_path = "/ccb/cybertron/smao10/openspliceai/data/toy/hg19.fa"
chromosome = "chr11"  # Replace with your chromosome name
window = 2000
flanking = 10000
reverse = True
start = 47364709 - (window//2) - (flanking//2)   # Replace with your start coordinate (1-based)
end = 47364709 + (window//2) + (flanking//2)   # Replace with your end coordinate (1-based)

# Extract sequence and print
sequence = extract_region(fasta_path, chromosome, start, end, reverse)
print(sequence[6000])
if sequence:
    output_file = "/ccb/cybertron/smao10/openspliceai/experiments/mutagenesis/experiment_1/data/mybpc3.fa"

    with open(output_file, "w") as file:
        file.write(f'>{chromosome}:{start}-{end}(-)_mybpc3\n')
        file.write(sequence)