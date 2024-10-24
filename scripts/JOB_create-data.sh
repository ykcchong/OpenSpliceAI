########################################
# Creating zebra_fish dataset
########################################
openspliceai create-data \
--genome-fasta /home/kchao10/data_ssalzbe1/khchao/ref_genome/zebra_fish/GCF_000002035.6_GRCz11_genomic.fna \
--annotation-gff /home/kchao10/data_ssalzbe1/khchao/ref_genome/zebra_fish/GCF_000002035.6_GRCz11_genomic.gff \
--output-dir /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_zebrafish/ \
--parse-type maximum

########################################
# Creating bee dataset
########################################
openspliceai create-data \
--genome-fasta /home/kchao10/data_ssalzbe1/khchao/ref_genome/bee/HAv3.1_genomic.fna \
--annotation-gff /home/kchao10/data_ssalzbe1/khchao/ref_genome/bee/HAv3.1_genomic.gff \
--output-dir /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_zebrafish/ \
--parse-type maximum

########################################
# Creating mouse dataset
########################################
openspliceai create-data \
--genome-fasta /home/kchao10/data_ssalzbe1/khchao/ref_genome/mouse/GCF_000001635.27_GRCm39_genomic.fna \
--annotation-gff /home/kchao10/data_ssalzbe1/khchao/ref_genome/mouse/GCF_000001635.27_GRCm39_genomic.gff \
--output-dir /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_mouse/ \
--parse-type maximum

########################################
# Creating arabadopsis dataset
########################################
openspliceai create-data \
--genome-fasta  /home/kchao10/data_ssalzbe1/khchao/ref_genome/arabadop/TAIR10.fna \
--annotation-gff /home/kchao10/data_ssalzbe1/khchao/ref_genome/arabadop/TAIR10.gff \
--output-dir ./arabadop/ \
--parse-type maximum

########################################
# Creating RefSeq dataset non-coding
########################################
openspliceai create-data \
--genome-fasta  /home/kchao10/data_ssalzbe1/khchao/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna \
--annotation-gff /home/kchao10/data_ssalzbe1/khchao/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.gff \
--output-dir /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_RefSeq_noncoding/ \
--parse-type maximum \
--biotype non-coding \
--chr-split train-test


########################################
# Creating MANE dataset
########################################
openspliceai create-data \
--genome-fasta  /home/kchao10/data_ssalzbe1/khchao/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna \
--annotation-gff /home/kchao10/data_ssalzbe1/khchao/ref_genome/homo_sapiens/MANE/v1.3/MANE.GRCh38.v1.3.refseq_genomic.gff \
--output-dir /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/ \
--parse-type maximum