# benchmarking
1. CPU time, GPU time
2. I/O reads and writes
3. Memory usage over time
4. GPU vs. CPU utilization
5. Copy volume

# commands
scalene [-options] variant_test.py [-program_args]

**NOTE: our implementation of variant supports both keras and pytorch models by default, no need to create separate script for that. 

## task
variant calling on Mills+1000G gold standard vcf file

## params
- flanking size: 80, 400, 2000, 10000
- pytorch vs. keras spliceai model

8 runs

# test run
 python ./experiments/benchmark_variant/variant_test.py -R /ccb/cybertron/smao10/openspliceai/data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna -A /ccb/cybertron/smao10/openspliceai/data/vcf/grch38.txt -f 80 -m /ccb/cybertron/smao10/openspliceai/models/spliceai-mane/80nt/model_80nt_rs42.pt -I /ccb/cybertron/smao10/openspliceai/data/vcf/hg38_mills1000g_subset10.vcf -O ./experiments/benchmark_variant/test/result.vcf

  python ./experiments/benchmark_variant/spliceai_orig.py -R /ccb/cybertron/smao10/openspliceai/data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna -A /ccb/cybertron/smao10/openspliceai/data/vcf/grch38.txt -f 80 -I /ccb/cybertron/smao10/openspliceai/data/vcf/hg38_mills1000g_subset10.vcf -O ./experiments/benchmark_variant/test/result.vcf