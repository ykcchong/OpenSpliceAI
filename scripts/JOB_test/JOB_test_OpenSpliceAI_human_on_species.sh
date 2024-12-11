#!/bin/bash
#SBATCH -N 1
#SBATCH --time=72:0:0
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH -A ssalzbe1_gpu
#SBATCH --array=0-15
#SBATCH --mem=64G  # Request 64 GB of memory

# Define the flanking sizes
FLANKING_SIZES=(80 400 2000 10000)

# Define the species
SPECIES=(mouse zebrafish honeybee arabidopsis)
# SPECIES=(MANE)

# Calculate the current species and flanking size based on the array task ID
SPECIES_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
FLANKING_INDEX=$((SLURM_ARRAY_TASK_ID % 4))

CURRENT_SPECIES=${SPECIES[$SPECIES_INDEX]}
FLANKING_SIZE=${FLANKING_SIZES[$FLANKING_INDEX]}

EXP_NUM=0
RANDOM_SEED=22

# # 1. train_spliceai_default_no_paralog_removal_outdir (MANE)
# OUTPUT_DIR="/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/test_outdir/test_human_model_on_species/${CURRENT_SPECIES}/flanking_${FLANKING_SIZE}"
# TEST_DATASET="/home/kchao10/data_ssalzbe1/khchao/data/spliceai_default_no_paralog_removed/train_test_dataset_${CURRENT_SPECIES}/dataset_test.h5"
# EXP_DIR=$OUTPUT_DIR/SpliceAI_${CURRENT_SPECIES}_train_${FLANKING_SIZE}_${EXP_NUM}_rs${RANDOM_SEED}
# PRETRAINED_MODEL="/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/train_spliceai_default_no_paralog_removal_outdir/MANE/flanking_${FLANKING_SIZE}/SpliceAI_MANE_train_${FLANKING_SIZE}_${EXP_NUM}_rs${RANDOM_SEED}/${EXP_NUM}/models/model_best.pt"
# LOG_DIR="TEST_LOG_${CURRENT_SPECIES}"

# 2. 
OUTPUT_DIR="/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/test_outdir/test_human_model_on_species/${CURRENT_SPECIES}/flanking_${FLANKING_SIZE}"
TEST_DATASET="/home/kchao10/data_ssalzbe1/khchao/data/spliceai_default_no_paralog_removed/train_test_dataset_${CURRENT_SPECIES}/dataset_test.h5"
EXP_DIR=$OUTPUT_DIR/SpliceAI_${CURRENT_SPECIES}_train_${FLANKING_SIZE}_${EXP_NUM}_rs${RANDOM_SEED}
PRETRAINED_MODEL="/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/train_outdir/SpliceAI-keras_data/MANE/flanking_${FLANKING_SIZE}/SpliceAI_MANE_train_${FLANKING_SIZE}_${EXP_NUM}_rs${RANDOM_SEED}/${EXP_NUM}/models/model_best.pt"
LOG_DIR="TEST_LOG_${CURRENT_SPECIES}_gencode_trained"

OUTPUT_FILE="$EXP_DIR/test_output_gencode_trained.log"
ERROR_FILE="$EXP_DIR/test_error_gencode_trained.log"

mkdir -p $OUTPUT_DIR
mkdir -p ${EXP_DIR}
openspliceai test --flanking-size $FLANKING_SIZE \
--test-dataset $TEST_DATASET \
--output-dir $OUTPUT_DIR \
--project-name ${CURRENT_SPECIES}_train \
--exp-num ${EXP_NUM} \
--pretrained-model ${PRETRAINED_MODEL} \
--random-seed ${RANDOM_SEED} \
--log-dir ${LOG_DIR} \
--loss cross_entropy_loss > $OUTPUT_FILE 2> $ERROR_FILE