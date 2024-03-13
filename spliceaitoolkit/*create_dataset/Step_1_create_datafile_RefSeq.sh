python Step_1_create_datafile.py \
--genome-fasta  /Users/chaokuan-hao/Documents/Projects/ref_genome/homo_sapiens/NCBI_Refseq_chr_fixed/GCF_000001405.40_GRCh38.p14_genomic.fna \
--annotation-gff /Users/chaokuan-hao/Documents/Projects/ref_genome/homo_sapiens/NCBI_Refseq_chr_fixed/GCF_000001405.40_GRCh38.p14_genomic.gff \
--output-dir /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/results/train_test_dataset_RefSeq/ \
--parse-type all_isoforms \
> Step_1_create_datafile_RefSeq.log 2> Step_1_create_datafile_RefSeq_error.log