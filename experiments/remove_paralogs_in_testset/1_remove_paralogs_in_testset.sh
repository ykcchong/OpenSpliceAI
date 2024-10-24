#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=72:0:0
#SBATCH --partition=parallel
#SBATCH -A ssalzbe1_gpu
#SBATCH --mem=64G
#SBATCH --array=0-4

# Define an array of species
SPECIES=(MANE mouse zebrafish honeybee arabidopsis)

# Get the species corresponding to the SLURM array index
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

echo "Running job for species: $species"
python 1_remove_paralogs_in_testset.py \
    --spliceai_keras_train_txt /home/kchao10/scr4_ssalzbe1/khchao/SpliceAI_train_code/Canonical/canonical_sequence.txt \
    --openspliceai_train_fasta /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_${species}/train.fa \
    --test_h5 /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_${species}/datafile_test.h5 \
    --output_dir ./${species}_data \
    --min_identity 0.8 \
    --min_coverage 0.5