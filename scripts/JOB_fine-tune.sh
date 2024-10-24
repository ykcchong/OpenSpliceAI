#!/bin/bash
#SBATCH -N 1
#SBATCH --time=48:0:0
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH -A ssalzbe1_gpu
#SBATCH --array=0-3
#SBATCH --mem=40G  # Request 64 GB of memory

# Define the flanking sizes
FLANKING_SIZES=(80 400 2000 10000)

# Get the current flanking size based on the array task ID
FLANKING_SIZE=${FLANKING_SIZES[$SLURM_ARRAY_TASK_ID]}

# Set up output directory
OUTPUT_DIR="../results/model_fine-tune_outdir/flanking_${FLANKING_SIZE}"
mkdir -p $OUTPUT_DIR
OUTPUT_FILE="$OUTPUT_DIR/output.log"
ERROR_FILE="$OUTPUT_DIR/error.log"

# Run the OpenSpliceAI fine-tuning command
openspliceai fine-tune --flanking-size $FLANKING_SIZE \
--exp-num $SLURM_ARRAY_TASK_ID \
--train-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_mouse/dataset_train.h5 \
--test-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_mouse/dataset_test.h5 \
--output-dir $OUTPUT_DIR \
--project-name human_mouse_fine-tune \
--random-seed 22 \
--pretrained-model /home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/models/spliceai-mane/${FLANKING_SIZE}nt/model_${FLANKING_SIZE}nt_rs12.pt \
--loss cross_entropy_loss > $OUTPUT_FILE 2> $ERROR_FILE
