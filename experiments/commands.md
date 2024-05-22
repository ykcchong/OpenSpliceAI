
# setup
python setup.py install

## predict
spliceai-toolkit predict -m models/spliceai-mane/400nt/model_400nt_rs40.pt -o results/predict -f 400 -i data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna

spliceai-toolkit predict -m models/spliceai-mane/400nt/model_400nt_rs40.pt -o results/predict -f 400 -i data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna -g data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.gff

## create-data
spliceai-toolkit create-data --output-dir results/create-data --genome-fasta data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14
_genomic.fna --annotation-gff data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.gff