#!/bin/bash
#SBATCH -N 1
#SBATCH --time=72:0:0
#SBATCH -p ica100
#SBATCH --gres=gpu:1
#SBATCH -A ssalzbe1_gpu
#SBATCH --array=0-19
#SBATCH --mem=64G  # Request 64 GB of memory

# Define the flanking sizes
FLANKING_SIZES=(80 400 2000 10000)
# FLANKING_SIZES=(80)
# Define the species
SPECIES=(MANE mouse honeybee zebrafish arabidopsis)
# SPECIES=(honeybee)

# Calculate the current species and flanking size based on the array task ID
SPECIES_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
FLANKING_INDEX=$((SLURM_ARRAY_TASK_ID % 4))

CURRENT_SPECIES=${SPECIES[$SPECIES_INDEX]}
FLANKING_SIZE=${FLANKING_SIZES[$FLANKING_INDEX]}
RANDOM_SEED=11

# Set up output directory
OUTPUT_DIR="/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/calibrate_outdir/${CURRENT_SPECIES}/flanking_${FLANKING_SIZE}"
mkdir -p $OUTPUT_DIR
OUTPUT_FILE="$OUTPUT_DIR/output.log"
ERROR_FILE="$OUTPUT_DIR/error.log"

# Set up dataset paths
TRAIN_DATASET="/home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_${CURRENT_SPECIES}/dataset_train.h5"
TEST_DATASET="/home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_${CURRENT_SPECIES}/dataset_test.h5"
CURRENT_SPECIES_lowercase="${CURRENT_SPECIES,,}"

TEMPERATURE_FILE="/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/calibrate_outdir/${CURRENT_SPECIES}/flanking_${FLANKING_SIZE}/temperature.pt"
PRETRAINED_MODEL="/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/models/spliceai-${CURRENT_SPECIES_lowercase}/${FLANKING_SIZE}nt/model_${FLANKING_SIZE}nt_rs${RANDOM_SEED}.pth"

echo "Current species: $CURRENT_SPECIES"
echo "Flanking size: $FLANKING_SIZE"
echo "Random seed: $RANDOM_SEED"
echo "Output directory: $OUTPUT_DIR"
echo "Train dataset: $TRAIN_DATASET"
echo "Test dataset: $TEST_DATASET"
echo "Pretrained model: $PRETRAINED_MODEL"

# Run the OpenSpliceAI calibrate command
openspliceai calibrate --flanking-size $FLANKING_SIZE \
--train-dataset $TRAIN_DATASET \
--test-dataset $TEST_DATASET \
--output-dir $OUTPUT_DIR \
--project-name human_${CURRENT_SPECIES}_calibrate \
--random-seed ${RANDOM_SEED} \
--pretrained-model $PRETRAINED_MODEL \
--loss cross_entropy_loss > $OUTPUT_FILE 2> $ERROR_FILE
