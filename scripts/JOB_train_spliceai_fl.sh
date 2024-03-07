#!/bin/bash
#SBATCH --job-name=JOB_spliceai_pytorch_train_MANE_small_dataset
#SBATCH --time=48:0:0
#SBATCH -p a100 # specify the GPU partition
#SBATCH -N 1 # Number of nodes
#SBATCH --ntasks-per-node=48
#SBATCH --gres=gpu:1
#SBATCH -A ssalzbe1_gpu ### Slurm-account is usually the PI’s userid
#SBATCH --export=ALL


# Load system modules
ml purge

source activate /home/kchao10/miniconda3/envs/pytorch_cuda

# Check if the Python interpreter picks up packages from both the system and the Conda environment
which python
python -c "import sys; print(sys.path)"

spliceai-toolkit train --flanking-size 80 \
--exp-num full_dataset_seq_gamma_2 \
--train-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/dataset_train.h5 \
--test-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/dataset_test.h5 \
--output-dir /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/results/model_train_outdir/ \
--project-name human_MANE_loss_test \
--model SpliceAI \
--loss focal_loss > train_splan_MANE_fl_gamma_2.log 2> train_splan_MANE_fl_gamma_2_error.log


# python train_splan.py --flanking-size 80 \
# --exp-num full_dataset_h5py_version \
# --training-target MANE \
# --train-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/dataset_train.h5 \
# --test-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/dataset_test.h5 \
# --project-root /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/ \
# --project-name SpliceAI_Human_MANE \
# --model LocalTransformer \
# > train_splan_LocalTransformer_MANE_80.log 2> train_splan_LocalTransformer_MANE_80_error.log