#!/bin/bash
#SBATCH --job-name=JOB_spliceai_pytorch_train_MANE_small_dataset
#SBATCH --time=48:0:0
#SBATCH -p a100 # specify the GPU partition
#SBATCH -N 1 # Number of nodes
#SBATCH --gres=gpu:1
#SBATCH -A ssalzbe1_gpu ### Slurm-account is usually the PI’s userid
#SBATCH --export=ALL


# Load system modules
ml purge

source activate /home/kchao10/miniconda3/envs/pytorch_cuda

# Check if the Python interpreter picks up packages from both the system and the Conda environment
which python
python -c "import sys; print(sys.path)"

FLANKING_SIZE=80
SPECIES=arabadop
RANDOPM_SEED=12
LOSS_FUNC=cross_entropy_loss
EXP_NUM=full_dataset

mkdir /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/results/model_train_outdir/SpliceAI_${LOSS_FUNC}_${SPECIES}_adeptive_lr_${FLANKING_SIZE}_${EXP_NUM}_rs${RANDOPM_SEED}/

echo "spliceai-toolkit train --flanking-size ${FLANKING_SIZE} \
--exp-num ${EXP_NUM} \
--train-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_${SPECIES}/dataset_train.h5 \
--test-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_${SPECIES}/dataset_test.h5 \
--output-dir /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/results/model_train_outdir/ \
--project-name ${SPECIES}_adeptive_lr \
--random-seed ${RANDOPM_SEED} \
--model SpliceAI \
--loss ${LOSS_FUNC} > /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/results/model_train_outdir/SpliceAI_${LOSS_FUNC}_${SPECIES}_adeptive_lr_${FLANKING_SIZE}_${EXP_NUM}_rs${RANDOPM_SEED}/train.log 2> /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/results/model_train_outdir/SpliceAI_${LOSS_FUNC}_${SPECIES}_adeptive_lr_${FLANKING_SIZE}_${EXP_NUM}_rs${RANDOPM_SEED}/train_error.log"