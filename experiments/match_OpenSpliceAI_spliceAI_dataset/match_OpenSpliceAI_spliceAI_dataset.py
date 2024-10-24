import csv

def read_openspliceai_genes(filename):
    genes = set()
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file, delimiter='\t')
        for row in csv_reader:
            if len(row) >= 4:
                gene = row[3]
                if gene.startswith("gene-"):
                    gene = gene[5:]  # Remove "gene-" prefix
                genes.add(gene)
    return genes

# def read_spliceai_keras_genes(filename):
#     genes = set()
#     selected_chromosomes = {'chr1', 'chr3', 'chr5', 'chr7', 'chr9'}
#     with open(filename, 'r') as file:
#         csv_reader = csv.reader(file, delimiter='\t')
#         for row in csv_reader:
#             if len(row) >= 3 and row[2] in selected_chromosomes:
#                 genes.add(row[0])
#     return genes

def read_spliceai_keras_genes(filename):
    genes = set()
    selected_chromosomes = {'chr1', 'chr3', 'chr5', 'chr7', 'chr9'}
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file, delimiter='\t')
        for row in csv_reader:
            if len(row) >= 3 and row[2] not in selected_chromosomes:
                genes.add(row[0])
    return genes

def check_other_chromosomes(filename, genes_to_check):
    genes_on_other_chr = set()
    other_chromosomes = set()
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file, delimiter='\t')
        for row in csv_reader:
            if len(row) >= 3 and row[0] in genes_to_check:
                if row[2] not in {'chr1', 'chr3', 'chr5', 'chr7', 'chr9'}:
                    genes_on_other_chr.add(row[0])
                    other_chromosomes.add(row[2])
    return genes_on_other_chr, other_chromosomes

def compare_genes(file1, file2):
    openspliceai_genes = read_openspliceai_genes(file1)
    spliceai_keras_genes = read_spliceai_keras_genes(file2)

    common_genes = openspliceai_genes.intersection(spliceai_keras_genes)
    only_in_openspliceai = openspliceai_genes - spliceai_keras_genes
    only_in_spliceai_keras = spliceai_keras_genes - openspliceai_genes

    # Check if genes only in OpenSpliceAI are on other chromosomes in SpliceAI Keras
    genes_on_other_chr, other_chromosomes = check_other_chromosomes(file2, only_in_openspliceai)

    print(f"Total genes in OpenSpliceAI: {len(openspliceai_genes)}")
    print(f"Total genes in SpliceAI Keras (chr1,3,5,7,9): {len(spliceai_keras_genes)}")
    print(f"Common genes: {len(common_genes)}")
    print(f"Genes only in OpenSpliceAI: {len(only_in_openspliceai)}")
    print(f"Genes only in SpliceAI Keras (chr1,3,5,7,9): {len(only_in_spliceai_keras)}")
    print(f"Genes from OpenSpliceAI found on other chromosomes in SpliceAI Keras: {len(genes_on_other_chr)}")
    print(f"Other chromosomes found: {', '.join(sorted(other_chromosomes))}")

    # Write results to files
    with open("common_genes.txt", "w") as f:
        for gene in common_genes:
            f.write(f"{gene}\n")

    with open("only_in_openspliceai.txt", "w") as f:
        for gene in only_in_openspliceai:
            f.write(f"{gene}\n")

    with open("only_in_spliceai_keras.txt", "w") as f:
        for gene in only_in_spliceai_keras:
            f.write(f"{gene}\n")

    with open("openspliceai_genes_on_other_chr.txt", "w") as f:
        for gene in genes_on_other_chr:
            f.write(f"{gene}\n")

if __name__ == "__main__":
    # OpenSpliceAI_fn = "/home/kchao10/data_ssalzbe1/khchao/data/spliceai_default_no_paralog_removed/train_test_dataset_MANE/stats.txt"
    OpenSpliceAI_fn = "/home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/train_stats.txt"
    SpliceAI_Keras_fn = "/home/kchao10/scr4_ssalzbe1/khchao/SpliceAI_train_code/Canonical/canonical_dataset.txt"

    compare_genes(OpenSpliceAI_fn, SpliceAI_Keras_fn)