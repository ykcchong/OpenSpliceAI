#!/bin/bash
#SBATCH -N 1
#SBATCH --time=72:0:0
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH -A ssalzbe1_gpu
#SBATCH --array=0-3
#SBATCH --mem=64G  # Request 64 GB of memory

# Define the flanking sizes
FLANKING_SIZES=(80 400 2000 10000)

# Define the species
# SPECIES=(MANE mouse zebrafish honeybee arabidopsis)
SPECIES=(MANE)

# Calculate the current species and flanking size based on the array task ID
SPECIES_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
FLANKING_INDEX=$((SLURM_ARRAY_TASK_ID % 4))

CURRENT_SPECIES=${SPECIES[$SPECIES_INDEX]}
FLANKING_SIZE=${FLANKING_SIZES[$FLANKING_INDEX]}

EXP_NUM=0
RANDOM_SEED=22

PRETRAINED_MODEL="/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/models/spliceai/SpliceAI_models/SpliceNet${FLANKING_SIZE}_c1.h5"

# # 1. Paralogs removed
# OUTPUT_DIR="../results/train_my_split_paralog_removal_outdir/${CURRENT_SPECIES}/flanking_${FLANKING_SIZE}"
# TEST_DATASET="/home/kchao10/data_ssalzbe1/khchao/data/paralog_removal/train_test_dataset_${CURRENT_SPECIES}/dataset_test.h5"
# EXP_DIR=$OUTPUT_DIR/SpliceAI_${CURRENT_SPECIES}_train_${FLANKING_SIZE}_${EXP_NUM}_rs${RANDOM_SEED}
# OUTPUT_FILE="$EXP_DIR/test_SpliceAI-Keras_output.log"
# ERROR_FILE="$EXP_DIR/test_SpliceAI-Keras_error.log"

# # 2. No paralogs removed
# OUTPUT_DIR="../results/train_my_split_no_paralog_removal_outdir/${CURRENT_SPECIES}/flanking_${FLANKING_SIZE}"
# TEST_DATASET="/home/kchao10/data_ssalzbe1/khchao/data/no_paralog_removal/train_test_dataset_${CURRENT_SPECIES}/dataset_test.h5"
# EXP_DIR=$OUTPUT_DIR/SpliceAI_${CURRENT_SPECIES}_train_${FLANKING_SIZE}_${EXP_NUM}_rs${RANDOM_SEED}
# OUTPUT_FILE="$EXP_DIR/test_SpliceAI-Keras_output_no_paralogs_removed.log"
# ERROR_FILE="$EXP_DIR/test_SpliceAI-Keras_error_no_paralogs_removed.log"


# # 4. SpliceAI default No paralogs removed
# OUTPUT_DIR="../results/train_spliceai_default_no_paralog_removal_outdir/${CURRENT_SPECIES}/flanking_${FLANKING_SIZE}"
# # Set up dataset paths (adjust these as needed for each species)
# TEST_DATASET="/home/kchao10/data_ssalzbe1/khchao/data/spliceai_default_no_paralog_removed/train_test_dataset_${CURRENT_SPECIES}/dataset_test.h5"
# EXP_DIR=$OUTPUT_DIR/SpliceAI_${CURRENT_SPECIES}_train_${FLANKING_SIZE}_${EXP_NUM}_rs${RANDOM_SEED}
# OUTPUT_FILE="$EXP_DIR/test_SpliceAI-Keras_spliceai_default_no_paralog_removed_output.log"
# ERROR_FILE="$EXP_DIR/test_SpliceAI-Keras_spliceai_default_no_paralog_removed_error.log"

# # 5. New model arch implementation
# OUTPUT_DIR="../results/train_outdir/new_model_arch_spliceai_default_paralog_removed/${CURRENT_SPECIES}/flanking_${FLANKING_SIZE}"
# # Set up dataset paths (adjust these as needed for each species)
# TEST_DATASET="/home/kchao10/data_ssalzbe1/khchao/data/spliceai_default_paralog_removed/train_test_dataset_${CURRENT_SPECIES}/dataset_test.h5"
# EXP_DIR=$OUTPUT_DIR/SpliceAI_${CURRENT_SPECIES}_train_${FLANKING_SIZE}_${EXP_NUM}_rs${RANDOM_SEED}
# OUTPUT_FILE="$EXP_DIR/test_SpliceAI-Keras_output.log"
# ERROR_FILE="$EXP_DIR/test_SpliceAI-Keras_error.log"

# # 6. SpliceAI-keras GENCODE dataset
# OUTPUT_DIR="../results/train_outdir/SpliceAI-keras_data/${CURRENT_SPECIES}/flanking_${FLANKING_SIZE}"
# # Set up dataset paths (adjust these as needed for each species)
# TEST_DATASET="/home/kchao10/data_ssalzbe1/khchao/data/prev_data/train_test_dataset_SpliceAI27/dataset_test.h5"
# EXP_DIR=$OUTPUT_DIR/SpliceAI_${CURRENT_SPECIES}_train_${FLANKING_SIZE}_${EXP_NUM}_rs${RANDOM_SEED}
# OUTPUT_FILE="$EXP_DIR/test_SpliceAI-Keras_output.log"
# ERROR_FILE="$EXP_DIR/test_SpliceAI-Keras_error.log"

# 7. SpliceAI-keras cleaned test dataset
OUTPUT_DIR="../results/test_outdir/clean_test_dataset/${CURRENT_SPECIES}/flanking_${FLANKING_SIZE}"
# Set up dataset paths (adjust these as needed for each species)
TEST_DATASET="/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/experiments/remove_paralogs_in_testset/filtered_test_dataset.h5"
EXP_DIR=$OUTPUT_DIR/SpliceAI_${CURRENT_SPECIES}_train_${FLANKING_SIZE}_${EXP_NUM}_rs${RANDOM_SEED}
OUTPUT_FILE="$EXP_DIR/test_SpliceAI-Keras_output.log"
ERROR_FILE="$EXP_DIR/test_SpliceAI-Keras_error.log"

# rm -rf ${EXP_DIR}/${RANDOM_SEED}/TEST_LOG_SpliceAI-Keras

echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "TEST_DATASET: $TEST_DATASET"
echo "EXP_DIR: $EXP_DIR"
echo "OUTPUT_FILE: $OUTPUT_FILE"
echo "ERROR_FILE: $ERROR_FILE"

# Run the OpenSpliceAI fine-tuning command
openspliceai test --flanking-size $FLANKING_SIZE \
--test-dataset $TEST_DATASET \
--output-dir $OUTPUT_DIR \
--project-name ${CURRENT_SPECIES}_train \
--exp-num ${EXP_NUM} \
--pretrained-model ${PRETRAINED_MODEL} \
--random-seed ${RANDOM_SEED} \
--test-target SpliceAI-Keras \
--loss cross_entropy_loss > $OUTPUT_FILE 2> $ERROR_FILE