
# setup
python setup.py install

## predict

### 1. full genome -> h5py file, predict on whole
spliceai-toolkit predict -m models/spliceai-mane/400nt/model_400nt_rs40.pt -o results/predict -f 400 -i data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna -t 0.9 -D > results/predict/SpliceAI_5000_400/output.log 2> results/predict/SpliceAI_5000_400/error.log

* fix: 
failing on step 2: this is because its converting the whole genome into the datafile, and each FASTA entry in this file is a whole chromosome -> too large, the batch size does not apply properly, running out of memory when converting the whole chromosome into an entry of the H5 file. 
- one way could be to split the chromosome into different pieces, need a way to detect that, and then create a new FASTA file which demarcates the different "pieces" of the chromosome. This time, need to make sure the pieces overlap by the flanking size, so that it predicts continuously on the whole chromosome.
- simpler way would just be to tell the user the genome is too large, need to make more specific entries in FASTA. 

### 2. full genome with full annotation -> h5py file, predicts on all genes
spliceai-toolkit predict -m models/spliceai-mane/400nt/model_400nt_rs40.pt -o results/predict -f 400 -i data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna -a data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.gff -t 0.9 -D > results/predict/SpliceAI_5000_400/output.log 2> results/predict/SpliceAI_5000_400/error.log

* fix:
failing on step 4: this is because it is running out of memory when saving the prediction results to the torch file (the stored array is too large)
- update get_prediction -> set a PERIODIC_SAVE_THRESHOLD_BATCHES variable, if reached this number of batches, periodically flush out the batch_ypred into the file, and load it again for next time. only execute like this if there are more thresholds than needed, and finish by setting the batch_ypred to None so that it is passed in as such to generate_bed
- update generate_bed -> since batch_ypred is no longer all the predictions, just don't pass it in (or pass in None) and let the function read it in from the torch file path

### 3. full genome with toy annotation -> h5py file, smaller file
spliceai-toolkit predict -m models/spliceai-mane/400nt/model_400nt_rs40.pt -o results/predict -f 400 -i data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna -a data/toy/human/test.gff -t 0.9 -D > results/predict/SpliceAI_5000_400/output.log 2> results/predict/SpliceAI_5000_400/error.log

* works

### 4. toy genome -> no h5py file, predict on whole 

### 5. toy genome with full annotation -> no h5py file, should discard coordinates outside of genome

### 6. toy genome with toy annotation (new model) -> no h5py file 

*NOTE: double check the BED file numbering of positions in generate_bed

## create-data
spliceai-toolkit create-data --output-dir results/create-data --genome-fasta data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14
_genomic.fna --annotation-gff data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.gff