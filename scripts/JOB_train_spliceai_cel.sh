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

spliceai-toolkit train --flanking-size 10000 \
--exp-num full_dataset \
--train-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE_test/dataset_train.h5 \
--test-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE_test/dataset_test.h5 \
--output-dir /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/results/model_train_outdir/ \
--project-name human_MANE_relabel \
--random-seed 15 \
--model SpliceAI \
--loss cross_entropy_loss > train_splan_MANE_relabel_10000_rs_15.log 2> train_splan_MANE_relabel_error_10000_rs_15.log

# spliceai-toolkit train --flanking-size 80 \
# --exp-num full_dataset_seq_w10 \
# --train-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/dataset_train.h5 \
# --test-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/dataset_test.h5 \
# --output-dir /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/results/model_train_outdir/ \
# --project-name human_MANE_loss_test \
# --model SpliceAI \
# --loss cross_entropy_loss > train_splan_MANE_cel_w10.log 2> train_splan_MANE_cel_w10_error.log


# python train_splan.py --flanking-size 80 \
# --exp-num full_dataset_h5py_version \
# --training-target MANE \
# --train-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/dataset_train.h5 \
# --test-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/dataset_test.h5 \
# --project-root /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/ \
# --project-name SpliceAI_Human_MANE \
# --model LocalTransformer \
# > train_splan_LocalTransformer_MANE_80.log 2> train_splan_LocalTransformer_MANE_80_error.log