# Define file paths
transcript_file_path = 'show_coords_paralogs.bed'
gff_file_path = '/home/kchao10/data_ssalzbe1/khchao/ref_genome/homo_sapiens/MANE/v1.3/MANE.GRCh38.v1.3.refseq_genomic.gff'
output_file_path = 'transcript_gene_mapping.txt'

# Read transcripts from the file
with open(transcript_file_path, 'r') as file:
    transcripts = [line.strip() for line in file]

# Function to parse GFF file and extract necessary info
def parse_gff(gff_path, transcripts):
    transcript_gene_map = {}
    with open(gff_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  # Skip header or comment lines
            parts = line.strip().split('\t')
            if parts[2] == 'mRNA':  # We're only interested in mRNA records
                attributes = parts[8]
                attr_dict = {item.split('=')[0]: item.split('=')[1] for item in attributes.split(';')}
                transcript_id = attr_dict.get('ID')
                if transcript_id in transcripts:
                    gene_id = attr_dict.get('Parent').replace('gene-', '')
                    chromosome = parts[0]
                    transcript_gene_map[transcript_id] = (gene_id, chromosome)
    return transcript_gene_map

# Parse the GFF file
transcript_gene_map = parse_gff(gff_file_path, transcripts)

# Write the results into a file
with open(output_file_path, 'w') as file:
    for transcript, (gene_id, chromosome) in transcript_gene_map.items():
        file.write(f'{transcript}\t{gene_id}\t{chromosome}\n')

print("Mapping completed and saved to", output_file_path)
