import re

def calculate_best_identity_and_coverage(sam_file, min_identity=0.9, min_coverage=0.8):
    # Dictionary to track the best alignment per query sequence
    best_alignments = {}

    # Define a regular expression to parse the CIGAR string
    cigar_pattern = re.compile(r'(\d+)([MIDNSHP=X])')
    
    with open(sam_file, 'r') as file:
        for line in file:
            # Skip header lines
            if line.startswith('@'):
                continue
            
            # Parse SAM fields
            fields = line.strip().split('\t')
            query_name = fields[0]  # Query sequence name
            cigar = fields[5]       # CIGAR string
            sequence = fields[9]    # Query sequence
            mapq = int(fields[4])   # Mapping quality
            
            # Calculate sequence length
            query_length = len(sequence)
            
            # Parse the CIGAR string to calculate aligned length and matches
            aligned_length = 0
            matches = 0
            for length, type in cigar_pattern.findall(cigar):
                length = int(length)
                if type in 'M=X':  # Matches (M), exact match (=), and mismatch (X)
                    aligned_length += length
                    matches += length
                elif type in 'I':  # Insertion to the reference
                    aligned_length += length
                elif type in 'D':  # Deletion from the reference
                    aligned_length += length
            
            # Calculate sequence identity and coverage
            identity = matches / aligned_length if aligned_length > 0 else 0
            coverage = aligned_length / query_length if query_length > 0 else 0
            
            # Store the best alignment based on identity and coverage
            if query_name not in best_alignments:
                # Store initial alignment
                best_alignments[query_name] = (identity, coverage)
            else:
                # Compare with the existing best alignment and update if better
                best_identity, best_coverage = best_alignments[query_name]
                if (identity > best_identity) or (identity == best_identity and coverage > best_coverage):
                    best_alignments[query_name] = (identity, coverage)
    
    # Count alignments that meet the threshold criteria
    similar_count = sum(1 for identity, coverage in best_alignments.values() 
                        if identity >= min_identity and coverage >= min_coverage)
    
    return similar_count

# Set thresholds for filtering
min_identity = 0.7  # Minimum sequence identity
min_coverage = 0.7  # Minimum coverage


# Count similar sequences in the training and testing SAM alignment files
input_dir = '/home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/'
train_alignment_file = f'{input_dir}minimap2/train_alignment.sam'
test_alignment_file = f'{input_dir}minimap2/test_alignment.sam'

similar_train_count = calculate_best_identity_and_coverage(train_alignment_file, min_identity, min_coverage)
similar_test_count = calculate_best_identity_and_coverage(test_alignment_file, min_identity, min_coverage)

print(f"Similar sequences in training alignment: {similar_train_count}")
print(f"Similar sequences in testing alignment: {similar_test_count}")
