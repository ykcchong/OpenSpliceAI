#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=24:0:0
#SBATCH --partition=parallel
#SBATCH -A ssalzbe1_gpu
#SBATCH --mem=64G
#SBATCH --array=0-4

annotation_gff=(
    "/home/kchao10/data_ssalzbe1/khchao/ref_genome/zebra_fish/GCF_000002035.6_GRCz11_genomic.gff"
    "/home/kchao10/data_ssalzbe1/khchao/ref_genome/bee/HAv3.1_genomic.gff"
    "/home/kchao10/data_ssalzbe1/khchao/ref_genome/mouse/GCF_000001635.27_GRCm39_genomic.gff"
    "/home/kchao10/data_ssalzbe1/khchao/ref_genome/arabadop/TAIR10.gff"
    "/home/kchao10/data_ssalzbe1/khchao/ref_genome/homo_sapiens/MANE/v1.3/MANE.GRCh38.v1.3.refseq_genomic.gff"
)

output_dir=(
    "/home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_zebrafish/"
    "/home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_honeybee/"
    "/home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_mouse/"
    "/home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_arabidopsis/"
    "/home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/"
)


# Use SLURM_ARRAY_TASK_ID to select the correct parameters
annotation=${annotation_gff[$SLURM_ARRAY_TASK_ID]}
output=${output_dir[$SLURM_ARRAY_TASK_ID]}
mkdir -p ${output}

# Run the command
splam extract $annotation -o ${output_dir}tmp_out_annotation -F feature_gene.txt > "$output/output.log" 2> "$output/error.log"