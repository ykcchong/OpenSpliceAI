#!/bin/bash
#SBATCH -N 1
#SBATCH --time=72:0:0
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH -A ssalzbe1_gpu
#SBATCH --array=0-4  # Adjust the array size to 5 since we're only running 5 experiments for arabidopsis
#SBATCH --mem=64G  # Request 64 GB of memory

# Define the flanking size (fixed to 10000)
FLANKING_SIZE=10000

# Define the species (fixed to arabidopsis)
SPECIES=arabidopsis

NUM_EXPERIMENTS=5

# Calculate the experiment number and random seed based on the array task ID
EXP_NUM=$((SLURM_ARRAY_TASK_ID % NUM_EXPERIMENTS))
RANDOM_SEED=$((10 + EXP_NUM))  # Adjust the random seed as needed

OUTPUT_DIR="/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/test_outdir/FINAL/${SPECIES}/flanking_${FLANKING_SIZE}"
TEST_DATASET="/home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_${SPECIES}/dataset_test.h5"
EXP_DIR=$OUTPUT_DIR/SpliceAI_${SPECIES}_train_${FLANKING_SIZE}_${EXP_NUM}_rs${RANDOM_SEED}
PRETRAINED_MODEL="/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/models/spliceai/SpliceAI_models/SpliceNet${FLANKING_SIZE}_c$((EXP_NUM+1)).h5"

LOG_DIR="TEST_LOG_SPLICEAI_KERAS"
OUTPUT_FILE="$EXP_DIR/test_splicai_keras_output.log"
ERROR_FILE="$EXP_DIR/test_splicai_keras_error.log"

mkdir -p ${EXP_DIR}

echo "Species: $SPECIES"
echo "Flanking size: $FLANKING_SIZE"
echo "Experiment number: $EXP_NUM"
echo "Random seed: $RANDOM_SEED"
echo "Output directory: $OUTPUT_DIR"
echo "Test dataset: $TEST_DATASET"
echo "Experiment directory: $EXP_DIR"
echo "Pretrained model: $PRETRAINED_MODEL"
echo "Log directory: $LOG_DIR"
echo "Output file: $OUTPUT_FILE"
echo "Error file: $ERROR_FILE"

openspliceai test --flanking-size $FLANKING_SIZE \
--test-dataset $TEST_DATASET \
--output-dir $OUTPUT_DIR \
--project-name ${SPECIES}_train \
--exp-num ${EXP_NUM} \
--pretrained-model ${PRETRAINED_MODEL} \
--random-seed ${RANDOM_SEED} \
--log-dir ${LOG_DIR} \
--test-target SpliceAI-Keras \
--loss cross_entropy_loss > $OUTPUT_FILE 2> $ERROR_FILE
