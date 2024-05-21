
# setup
python setup.py install
navigate to spliceaitoolkit directory

## predict
spliceai-toolkit predict -m ../models/spliceai-mane/400nt/model_400nt_rs40.pt -o ../results/predict -f 400 -i ../data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna
