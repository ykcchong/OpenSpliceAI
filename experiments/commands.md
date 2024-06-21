
# setup
python setup.py install

# predict

## updates (since last working model)
1. use h5 files for storing predictions
- previously used pt, but found that if too many predictions saved, would run out of memory -> needed a way to flush predictions to file continuously, pt does not support but h5 does
2. 4o option
- faster method which combines prediction and bed file writing together, this reduces the stored memory size, only extracting the necessary donors and acceptors above the threshold
- is now the default method, but 4 + 5 possible
3. tried multithreading -> DOES NOT WORK -> removed


## questions for kh
1. is multithreading implemented in python, or will it be easier to do in C? will this command-line utility eventually be converted to Cpython like in Splam? it would provide a significant speedup in C implementation, and multithreading will definitely make predit run a lot faster (it is also inherently parallelizeable)
    - if implementing in Python, should I use ThreadPoolExecutor, or is there a better way to do this? double-check logic
2. as predict is developed, there are always features that I feel like adding to improve runtime/memory usage, but it may vary for different users so i always keep options to tune parameters, it is better practice to infer these features from the user's system/automatically attempt the most optimal running parameters, or should i keep everything as a manual parameter?

## questions for steven and ela
1. for the application side, what do you think about packaging this toolkit alongside Splam? 

## 1. full genome -> h5py file, predict on whole
spliceai-toolkit predict -m models/spliceai-mane/400nt/model_400nt_rs40.pt -o results/predict -f 400 -i data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna -t 0.9 -D > results/predict/SpliceAI_5000_400/output.log 2> results/predict/SpliceAI_5000_400/error.log

### testing notes:
[x] failing (out of memory) on step 2: this is because its converting the whole genome into the datafile, and each FASTA entry in this file is a whole chromosome -> too large, the batch size does not apply properly, running out of memory when converting the whole chromosome into an entry of the H5 file. -> implemented the splitting
- one way could be to split the chromosome into different pieces, need a way to detect that, and then create a new FASTA file which demarcates the different "pieces" of the chromosome. This time, need to make sure the pieces overlap by the flanking size, so that it predicts continuously on the whole chromosome. 
- simpler way would just be to tell the user the genome is too large, need to make more specific entries in FASTA. 

[x] when splitting the chromosome/long genes, the prediction will pad the flanking of the last chunk with N's when in reality there is more context that exists there. 
- workaround would be to add the flanking sequence onto the sequence (so each chunk is SPLIT_FASTA_THRESHOLD + CL_max length), giving slight overlap between chunks
- *NOTE: would need to ensure that SPLIT_FASTA_THRESHOLD is a multiple of the SL (sequence length, 5k) so that every chunk aligns perfectly with the boundary of the split*

## 2. full genome with full annotation -> h5py file, predicts on all genes
spliceai-toolkit predict -m models/spliceai-mane/400nt/model_400nt_rs40.pt -o results/predict -f 400 -i data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna -a data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.gff -t 0.9 -D > results/predict/SpliceAI_5000_400/output.log 2> results/predict/SpliceAI_5000_400/error.log

**with 8 threads**
spliceai-toolkit predict -m models/spliceai-mane/400nt/model_400nt_rs40.pt -o results/predict -f 400 -i data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna -a data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.gff -t 0.9 -@ 8 -D > results/predict/SpliceAI_5000_400/output.log 2> results/predict/SpliceAI_5000_400/error.log

### testing notes:
[x] failing on step 4: this is because it is running out of memory when saving the prediction results to the torch file (the stored array is too large)
- update get_prediction -> set a PERIODIC_SAVE_THRESHOLD_BATCHES variable, if reached this number of batches, periodically flush out the batch_ypred into the file, and load it again for next time. only execute like this if there are more thresholds than needed, and finish by setting the batch_ypred to None so that it is passed in as such to generate_bed
- update generate_bed -> since batch_ypred is no longer all the predictions, just don't pass it in (or pass in None) and let the function read it in from the torch file path

[x] new issue? failing on step 4: running out of memory? happens at different parts of get_prediction each time
- original memory issue is solved as the array is getting properly flushed out each time
- observed a huge spike in memory usage, almost all cpus used and cutting into swap memory, right before killing
- think this is due to the flushing to torch file, still needs to loads the entire torch file into memory before saving, which can cause issues, may need to come up with a different way to save predictions

solutions: 
1. save predictions into separate pt files, rather than trying to concatenate everything to the same one
2. extract all useful information from predictions before saving into pt file, reducing file size -> but not as useful if needed for other applications
3. find a different file format that can handle appending to files rather than reloading and recompressing... h5 files

[x] went with method 3 -> new issue, the prediction h5 file is 255.4 GIGABYTES
- will try out method 2, need to dynamically detect this?
    - instead, made flag with option to write info to file, otherwise will default to just extracting predictions without intermediate prediction file

[x] new method works -> is very slow specifically in BED file writing
- use ThreadPoolExecutor to multithread the BED file writing process 
    - do in both the generate_bed and extract_predictions functions
    - maybe in the future figure out if its possible with get_prediction
    - *NOTE: need to determine whether multithreading screws up order of BED file, esp LEN variable in convert_sequences*

[x] there is actually slowdown when multithreading convert_sequences...
- on the upside, threading automatically sorts all sequences in order
- will try a different multithreading method
- actually just got rid of threads altogether

[x] errors detected (not present in 3 for some reason), seems to have overinflated prediction numbers, as well as wrong coordinates for essentially all of the genes
- likely not from multithreading as issue was present before it was implemented 
- i think its an issue with the batching process -> it was

## 3. full genome with toy annotation -> h5py file, smaller file
spliceai-toolkit predict -m models/spliceai-mane/400nt/model_400nt_rs40.pt -o results/predict -f 400 -i data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna -a data/toy/human/test.gff -t 0.9 -D -p > results/predict/SpliceAI_5000_400/output.log 2> results/predict/SpliceAI_5000_400/error.log

spliceai-toolkit predict -m models/spliceai-mane/400nt/model_400nt_rs40.pt -o results/predict -f 400 -i data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna -a data/toy/human/test.gff -t 0.9 -D > results/predict/SpliceAI_5000_400/output.log 2> results/predict/SpliceAI_5000_400/error.log

**with 8 threads**
spliceai-toolkit predict -m models/spliceai-mane/400nt/model_400nt_rs40.pt -o results/predict -f 400 -i data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna -a data/toy/human/test.gff -t 0.9 -@ 8 -D -p > results/predict/SpliceAI_5000_400/output.log 2> results/predict/SpliceAI_5000_400/error.log

### testing notes (originally worked, broke after some improvements)
[x] benchmarks find that convert_sequences multithreading is slower
- undo multithreading

[x] file corruption during bed writing with 4 and 5, although marginal speedup observed. 
- i think cause is due to the multithreading... multiple attempts to write to file at the same time
- NO MORE MULTITHREADING GET RID OF IT AHHHH

[x] still incorrect coordinates with 4o, issue is not observed in 4 and 5 -> note that the order of genes is different, but same number of predictions generated as correct run
- probably due to some way the gene information is loaded (ordering during storage?)
- fixed by parsing through each batch (can contain multiple genes, genes can span multiple batches)

## 4. toy genome -> no h5py file, predict on whole 

## 5. toy genome with full annotation -> no h5py file, should discard coordinates outside of genome

## 6. toy genome with toy annotation (new model) -> no h5py file 

*NOTE: double check the BED file numbering of positions in generate_bed

# create-data
spliceai-toolkit create-data --output-dir results/create-data --genome-fasta data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14
_genomic.fna --annotation-gff data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.gff


# variant

## updates
1. added conversion from pt model to keras model to ensure compatibility
    - is this an issue for downstream? the predict function we make is built off pt, so it could work but will need to entirely rework variant

## standard test
spliceai-toolkit variant -m models/spliceai-mane/400nt/model_400nt_rs42.pt -f 400 -I data/vcf/input.vcf -O results/variant/output.vcf -R data/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna -A data/grch38.txt -D 100 -M 1