
# setup
python setup.py install

## predict

### 1. full genome -> h5py file, predict on whole
spliceai-toolkit predict -m models/spliceai-mane/400nt/model_400nt_rs40.pt -o results/predict -f 400 -i data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna -t 0.9

### 2. full genome with full annotation -> h5py file, predicts on all genes
spliceai-toolkit predict -m models/spliceai-mane/400nt/model_400nt_rs40.pt -o results/predict -f 400 -i data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna -a data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.gff -t 0.9

### 3. full genome with toy annotation -> h5py file, smaller file
spliceai-toolkit predict -m models/spliceai-mane/400nt/model_400nt_rs40.pt -o results/predict -f 400 -i data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna -a data/toy/human/test.gff -t 0.9

### 4. toy genome -> no h5py file, predict on whole 

### 5. toy genome with full annotation -> no h5py file, should discard coordinates outside of genome

### 6. toy genome with toy annotation (new model) -> no h5py file 

*NOTE: double check the BED file numbering of positions in generate_bed

## create-data
spliceai-toolkit create-data --output-dir results/create-data --genome-fasta data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14
_genomic.fna --annotation-gff data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.gff