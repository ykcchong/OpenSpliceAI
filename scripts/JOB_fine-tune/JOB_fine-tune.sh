#!/bin/bash
#SBATCH -N 1
#SBATCH --time=72:0:0
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH -A ssalzbe1_gpu
#SBATCH --array=0-79  
#SBATCH --mem=64G  # Request 64 GB of memory
#SBATCH --mail-type=end
#SBATCH --mail-user=kuanhao.chao@gmail.com

# Define the flanking sizes
FLANKING_SIZES=(80 400 2000 10000)

# Define the species
SPECIES=(mouse honeybee zebrafish arabidopsis)

# Calculate the total number of experiments per combination of species and flanking size
NUM_EXPERIMENTS=5

# Calculate the current species, flanking size, experiment number, and random seed based on the array task ID
SPECIES_INDEX=$((SLURM_ARRAY_TASK_ID / (NUM_EXPERIMENTS * ${#FLANKING_SIZES[@]})))
FLANKING_INDEX=$(((SLURM_ARRAY_TASK_ID / NUM_EXPERIMENTS) % ${#FLANKING_SIZES[@]}))
EXP_NUM=$((SLURM_ARRAY_TASK_ID % NUM_EXPERIMENTS))
RANDOM_SEED=$((10 + EXP_NUM))  # Adjust the random seed as needed

CURRENT_SPECIES=${SPECIES[$SPECIES_INDEX]}
FLANKING_SIZE=${FLANKING_SIZES[$FLANKING_INDEX]}

# Set up output directory
OUTPUT_DIR="/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/fine-tune_outdir/${CURRENT_SPECIES}/flanking_${FLANKING_SIZE}"
PROJECT_DIR="${OUTPUT_DIR}/SpliceAI_human_${CURRENT_SPECIES}_fine-tune_${FLANKING_SIZE}_${EXP_NUM}_rs${RANDOM_SEED}"
mkdir -p $PROJECT_DIR
OUTPUT_FILE="${PROJECT_DIR}/output.log"
ERROR_FILE="${PROJECT_DIR}/error.log"

# Set up dataset paths
TRAIN_DATASET="/home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_${CURRENT_SPECIES}/dataset_train.h5"
TEST_DATASET="/home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_${CURRENT_SPECIES}/dataset_test.h5"

PRETRAINED_MODEL="/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/models/spliceai-mane/${FLANKING_SIZE}nt/model_${FLANKING_SIZE}nt_rs${RANDOM_SEED}.pth"

echo "Starting job on $(hostname) at $(date)"   
echo "============================"
echo "Species: $CURRENT_SPECIES"
echo "Flanking size: $FLANKING_SIZE"
echo "Experiment number: $EXP_NUM"
echo "Random seed: $RANDOM_SEED"
echo "============================"

# Run the OpenSpliceAI fine-tuning command
openspliceai fine-tune --flanking-size $FLANKING_SIZE \
--train-dataset $TRAIN_DATASET \
--test-dataset $TEST_DATASET \
--output-dir $OUTPUT_DIR \
--project-name human_${CURRENT_SPECIES}_fine-tune \
--exp-num ${EXP_NUM} \
--random-seed ${RANDOM_SEED} \
--pretrained-model $PRETRAINED_MODEL \
--epochs 10 \
--scheduler CosineAnnealingWarmRestarts \
--loss cross_entropy_loss > $OUTPUT_FILE 2> $ERROR_FILE