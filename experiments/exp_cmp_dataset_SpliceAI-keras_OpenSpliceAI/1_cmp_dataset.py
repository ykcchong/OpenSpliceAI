# Step 3: Define the Python script to process these files
def load_sequences(filename):
    sequences = set()
    with open(filename, 'r') as file:
        for line in file:
            # Extract the sequence part of the entry
            sequence = line.split()[1].upper()
            sequences.add(sequence)
    return sequences

def load_fasta_sequences(filename):
    sequences = set()
    with open(filename, 'r') as file:
        sequence = ''
        for line in file:
            if line.startswith('>'):
                if sequence:
                    sequences.add(sequence)
                sequence = ''
            else:
                sequence += line.strip().upper()
        if sequence:
            sequences.add(sequence.upper())
    return sequences

def convert_to_fasta(input_txt, output_fasta):
    with open(input_txt, 'r') as infile, open(output_fasta, 'w') as outfile:
        for i, line in enumerate(infile):
            # Extract the chromosome and sequence
            chrom = line.split()[0].replace(':', '_')  # Replace ':' to avoid issues in headers
            sequence = line.split()[1]
            # Write to FASTA format
            outfile.write(f">{chrom}_{i}\n{sequence}\n")

def main():
    # Step 4: File paths
    spliceai_keras_dir='/home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/SpliceAI-Gencode/'
    canonical_sequence_path = f'{spliceai_keras_dir}canonical_sequence.txt'
    train_fa_path = '/home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/train.fa'
    test_fa_path = '/home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/test.fa'
    
    # Step 5: Split `canonical_sequence.txt` into training and testing
    with open(canonical_sequence_path, 'r') as infile, \
         open(f'{spliceai_keras_dir}spliceai_train.txt', 'w') as trainfile, \
         open(f'{spliceai_keras_dir}spliceai_test.txt', 'w') as testfile:
        
        for line in infile:
            chrom = line.split(':')[0]
            if chrom in ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']:
                testfile.write(line)
            else:
                trainfile.write(line)
    
    # Convert `spliceai_train.txt` to `spliceai_train.fa`
    convert_to_fasta(f'{spliceai_keras_dir}spliceai_train.txt', f'{spliceai_keras_dir}spliceai_train.fa')

    # Convert `spliceai_test.txt` to `spliceai_test.fa`
    convert_to_fasta(f'{spliceai_keras_dir}spliceai_test.txt', f'{spliceai_keras_dir}spliceai_test.fa')

    # # Load sequences from each file
    # spliceai_train_seqs = load_sequences(f'{spliceai_keras_dir}spliceai_train.txt')
    # spliceai_test_seqs = load_sequences(f'{spliceai_keras_dir}spliceai_test.txt')
    # openspliceai_train_seqs = load_fasta_sequences(train_fa_path)
    # openspliceai_test_seqs = load_fasta_sequences(test_fa_path)
    # print(f"SpliceAI training sequences: {len(spliceai_train_seqs)}")
    # print(f"SpliceAI testing sequences: {len(spliceai_test_seqs)}")
    # print(f"OpenSpliceAI training sequences: {len(openspliceai_train_seqs)}")
    # print(f"OpenSpliceAI testing sequences: {len(openspliceai_test_seqs)}")

    
    # # Compare the sequences
    # common_train_seqs = spliceai_train_seqs & openspliceai_train_seqs
    # common_test_seqs = spliceai_test_seqs & openspliceai_test_seqs

    # unique_spliceai_train = spliceai_train_seqs - openspliceai_train_seqs
    # unique_spliceai_test = spliceai_test_seqs - openspliceai_test_seqs
    # unique_openspliceai_train = openspliceai_train_seqs - spliceai_train_seqs
    # unique_openspliceai_test = openspliceai_test_seqs - spliceai_test_seqs

    # # Display results
    # results = {
    #     "Common training sequences": common_train_seqs,
    #     "Common testing sequences": common_test_seqs,
    #     "Unique SpliceAI training sequences": unique_spliceai_train,
    #     "Unique SpliceAI testing sequences": unique_spliceai_test,
    #     "Unique OpenSpliceAI training sequences": unique_openspliceai_train,
    #     "Unique OpenSpliceAI testing sequences": unique_openspliceai_test,
    # }
    results = {}
    return results

# Step 6: Run the script and get the results
results = main()
for k,v in results.items():
    print(f"{k}: {len(v)}")