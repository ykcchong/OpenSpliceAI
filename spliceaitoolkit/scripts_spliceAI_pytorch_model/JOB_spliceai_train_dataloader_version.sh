#!/bin/bash
#SBATCH --job-name=JOB_spliceai_pytorch_train_MANE_small_dataset
#SBATCH --time=48:0:0
#SBATCH -p a100 # specify the GPU partition
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=48
#SBATCH --gres=gpu:4
#SBATCH -A ssalzbe1_gpu ### Slurm-account is usually the PI’s userid
#SBATCH --export=ALL


# Load system modules
ml purge
module load gcc/9.3.0
module load cuda/11.1.0
module load anaconda

source activate /home/kchao10/miniconda3/envs/pytorch_cuda

# Check if the Python interpreter picks up packages from both the system and the Conda environment
which python
python -c "import sys; print(sys.path)"


python train_splan.py --flanking-size 2000 \
--exp-num full_dataset_h5py_version \
--training-target SpliceAI27 \
--train-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_SpliceAI27/dataset_train.h5 \
--test-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_SpliceAI27/dataset_test.h5 \
--project-root /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/ \
--project-name SpliceAI_Human_Genocode_hg19 \
--model SpliceAI \
> train_splan_Genocode_hg19_2000.log 2> train_splan_Genocode_hg19_2000_error.log

# --dataset-shuffle \