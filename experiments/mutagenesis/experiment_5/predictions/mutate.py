def mutate_fasta(input_file, mutation_position, mutation_base):
    # Read the input FASTA file
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Extract the sequence from the FASTA file
    header = lines[0].strip()
    sequence = ''.join(line.strip() for line in lines[1:])

    # Perform the mutation
    print(sequence[mutation_position], mutation_base)
    mutated_sequence = sequence[:mutation_position] + mutation_base + sequence[mutation_position + 1:]

    # Create the output file name
    output_file = input_file.replace('.fa', f'_mutated_{mutation_position}_{mutation_base}.fa')

    # Write the mutated sequence to the new FASTA file
    with open(output_file, 'w') as file:
        file.write(header + '\n')
        file.write(mutated_sequence + '\n')

# Example usage
input_fasta = '/ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/d/data/tp53_grch38.fa'
flanking = 11000
seqlen = 7687490-7668421
mutation_pos = 14147 + (flanking//2)
mutation_base = 'A' # original: 'G'
mutate_fasta(input_fasta, mutation_pos, mutation_base)