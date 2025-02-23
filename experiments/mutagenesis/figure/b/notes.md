# Data
100 randomly sampled donor and 100 randomly sampled acceptor sequences from the testing dataset (chr1, 3, 5, 7, 9). All sequences are 400 bp with the donor/acceptor motif located at the 199-200 indices, with the additional 10k flanking sequence (5k on each side). The sequences are appropriately cropped depending on the model used. Results are taken as the average of the scores between all samples

# Method
Mutate each base individually in the 400bp sequence to every other base. For each mutated sequence, input to the model and collect the predicted probabilities for the donor and acceptor sites (scored at the location of the site). Then, create a positional weight matrix for both donor and acceptor sites, for each base and position in the sequence where the weight reflects the change in probability of a donor/acceptor site for that given position and mutation (reference - mutation).

# Visualization
We visualize the PWM in a DNA logo of the reference sequence for both donor and acceptor sites. Note that the donor and acceptor sites are located in the middle of the sequence, where the largest spikes occur, reflecting their relative highest importance in the model. 

## Keras Note
Because Keras model takes a long time to run (on the order of weeks), distributed the job in parallel to 10 subprocesses on a cloud GPU compute server, Rockfish. Each was processed in batch for 10 sequences and the results were aggregated. 

## Usage
run_keras_mutagenesis.py -> SLURM script for running the keras mutagenesis
    -> keras_mutagenesis.py -> actual script
keras_job_visualize.py -> visualization for the keras results (which are stored in separate jobs)
    -> vis.py -> also runs on this generate_dna_logo

pytorch_mut.py -> pytorch mutagenesis module for running and visualization of pytorch results
    -> vis.py -> visualization utils