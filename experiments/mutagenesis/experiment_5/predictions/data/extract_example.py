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
    

# MYBPC3 gene coords: chr11:47,351,957-47,375,253(-) (hg19)
# 47364709 (mut pos)
# name = 'mybpc3'
# fasta_path = "/ccb/cybertron/smao10/openspliceai/data/toy/hg19.fa"
# chromosome = "chr11" 
# start = 47351957
# end = 47375253
# print(end-start)
# reverse = True

# # TP53 gene coords: chr17:7,571,739-7,590,808(-) (hg19)
# name = 'tp53'
# fasta_path = "/ccb/cybertron/smao10/openspliceai/data/toy/hg19.fa"
# chromosome = "chr17" 
# start = 7571739
# end = 7590808
# print(end-start)
# reverse = True


# TP53 gene coords: chr17:7,668,421-7,687,490(-) (GRCh38 MANE)
name = 'tp53_grch38_small'
fasta_path = "/ccb/cybertron2/smao10/openspliceai/data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna"
chromosome = "chr17" 
start = 7668421
end = 7687490
print('len', end-start)
reverse = True

window = 100
flanking = 0
extract_start = start - (window // 2) - (flanking // 2) 
extract_end = end + (window // 2) + (flanking // 2) 

# Extract sequence and print
sequence = extract_region(fasta_path, chromosome, extract_start, extract_end, reverse)
if sequence:
    output_file = f"/ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/d/data/{name}.fa"

    with open(output_file, "w") as file:
        file.write(f'>{chromosome}:{extract_start}-{extract_end}(-)_{name}_{window}window_{flanking}flank\n')
        file.write(sequence)