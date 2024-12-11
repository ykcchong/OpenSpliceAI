#!/bin/bash
#SBATCH -N 1
#SBATCH --time=72:0:0
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH -A ssalzbe1_gpu
#SBATCH --array=0
#SBATCH --mem=64G  # Request 64 GB of memory

# Set the flanking size to 10000
FLANKING_SIZE=10000

# Set the species to arabidopsis
SPECIES=arabidopsis

NUM_EXPERIMENTS=1

# Calculate the experiment number and random seed based on the array task ID
# EXP_NUM=$((SLURM_ARRAY_TASK_ID % NUM_EXPERIMENTS))
EXP_NUM=4
RANDOM_SEED=$((10 + EXP_NUM))  # Adjust the random seed as needed

OUTPUT_DIR="/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/test_outdir/FINAL/${SPECIES}/flanking_${FLANKING_SIZE}"
TEST_DATASET="/home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_${SPECIES}/dataset_test.h5"
EXP_DIR=$OUTPUT_DIR/SpliceAI_${SPECIES}_train_${FLANKING_SIZE}_${EXP_NUM}_rs${RANDOM_SEED}
PRETRAINED_MODEL="/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/train_outdir/FINAL/${SPECIES}/flanking_${FLANKING_SIZE}/SpliceAI_${SPECIES}_train_${FLANKING_SIZE}_${EXP_NUM}_rs${RANDOM_SEED}/${EXP_NUM}/models/model_best.pt"

LOG_DIR="TEST_LOG_OPENSPLICEAI_${SPECIES}"
OUTPUT_FILE="$EXP_DIR/test_opensplicai_${SPECIES}_output.log"
ERROR_FILE="$EXP_DIR/test_opensplicai_${SPECIES}_error.log"

echo "Current species: $SPECIES"    
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

mkdir -p ${EXP_DIR}
# rm -rf ${EXP_DIR}/${EXP_NUM}/${LOG_DIR}/

openspliceai test --flanking-size $FLANKING_SIZE \
--test-dataset $TEST_DATASET \
--output-dir $OUTPUT_DIR \
--project-name ${SPECIES}_train \
--exp-num ${EXP_NUM} \
--pretrained-model ${PRETRAINED_MODEL} \
--random-seed ${RANDOM_SEED} \
--log-dir ${LOG_DIR} \
--test-target "OpenSpliceAI" \
--loss cross_entropy_loss > $OUTPUT_FILE 2> $ERROR_FILE
