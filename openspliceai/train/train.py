"""
train.py

Implements the training, validation, and testing procedures for the SpliceAI model. 
- Setup of computational device based on system capabilities (CUDA, MPS, CPU).
- Initialization of model architecture, loss function, optimizer, and learning rate scheduler.
- Data loading and preprocessing to format genomic sequences and their labels for model consumption.
- Implementation of training and validation loops with metrics calculation and logging.
- Capability to save model checkpoints after each epoch for later analysis or inference.
- Integration with Weights & Biases for online tracking and visualization of training metrics.

Usage:
    For detailed usage and available options, run the script with the `-h` or `--help` flag.

Example:
    python train.py --output_dir=./outputs --project_name="SpliceAI" --flanking_size=200 --exp_num=1
                    --model="SpliceAI" --loss="cross_entropy" --train_dataset="./data/train.h5"
                    --test_dataset="./data/test.h5" --enable_wandb=False

Functions:
- `setup_device()`: Determines the best computational device (CUDA, MPS, CPU) available for training.
- `initialize_paths()`: Sets up directories for saving outputs, including model checkpoints and logs.
- `initialize_model_and_optim()`: Initializes the SpliceAI model, along with its optimizer and learning rate scheduler.
- `load_data_from_shard()`: Loads and preprocesses data from a specified shard of the dataset, preparing it for the model.
- `train_epoch()`: Conducts a single epoch of training, including forward passes, backpropagation, and parameter updates.
- `valid_epoch()`: Evaluates the model on a validation or test dataset without updating model parameters.
- `model_evaluation()`: Calculates and logs various performance metrics during the training and validation phases.
- `train()`: Orchestrates the entire training process, leveraging the above functions to train and evaluate the model.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import platform
from openspliceai.train_base.spliceai import *
from openspliceai.train_base.utils import *
from openspliceai.constants import *
import time
import wandb # weights and biases: need to connect to this one

def initialize_model_and_optim(device, flanking_size):
    """
    Initializes the SpliceAI model, criterion (loss function), optimizer, and learning rate scheduler, 
    based on the provided flanking size and model architecture. 

    Parameters:
    - device (torch.device): The computational device to use (CUDA, MPS, or CPU).
    - flanking_size (int): The size of the flanking sequences, influencing the model architecture.

    Returns:
    - model (torch.nn.Module): The initialized SpliceAI model.
    - criterion (torch.nn.Module): The loss function to be used during training.
    - optimizer (torch.optim.Optimizer): The optimizer for training the model.
    - scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
    - params (dict): Dictionary containing model and training parameters.
    """
    # Hyper-parameters:
    # L: Number of convolution kernels
    # W: Convolution window size in each residual unit
    # AR: Atrous rate in each residual unit
    L = 32
    N_GPUS = 2
    W = np.asarray([11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1])
    BATCH_SIZE = 18*N_GPUS
    if int(flanking_size) == 80:
        W = np.asarray([11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1])
        BATCH_SIZE = 18*N_GPUS
    elif int(flanking_size) == 400:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
        BATCH_SIZE = 18*N_GPUS
    elif int(flanking_size) == 2000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10])
        BATCH_SIZE = 12*N_GPUS
    elif int(flanking_size) == 10000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21, 41, 41, 41, 41])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10, 25, 25, 25, 25])
        BATCH_SIZE = 6*N_GPUS
    CL = 2 * np.sum(AR*(W-1))
    print("\033[1mContext nucleotides: %d\033[0m" % (CL))
    print("\033[1mSequence length (output): %d\033[0m" % (SL))
    model = SpliceAI(L, W, AR).to(device)
    print(model, file=sys.stderr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, 1000, train_size * EPOCH_NUM)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 7, 8, 9], gamma=0.5)
    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'N_GPUS': N_GPUS}
    return model, optimizer, scheduler, params


def train(args):
    """
    Main function to train, validate, and test the SpliceAI model according to specified arguments.

    This function orchestrates the entire process of training the SpliceAI model, including loading the dataset,
    initializing the model and its components (optimizer, scheduler, etc.), and executing the training, validation,
    and testing loops. It handles logging of metrics and saves model checkpoints at each epoch.

    Parameters:
    - args (argparse.Namespace): Command-line arguments required for training
        - output_dir (str): The directory where output files and model checkpoints will be saved.
        - project_name (str): Name of the project, used for organizing outputs.
        - flanking_size (int): Size of the flanking sequences around splice sites.
        - exp_num (int): Experiment number to differentiate between different runs.
        - model (str): Model architecture to use.
        - loss (str): Loss function for training the model.
        - train_dataset (str): Path to the training dataset file.
        - test_dataset (str): Path to the testing dataset file.
        - enable_wandb (bool): Flag to disable logging to Weights & Biases.
    """
    print("Running OpenSpliceAI with 'train' mode")
    # assert training_target in ["RefSeq", "MANE", "SpliceAI", "SpliceAI27"]
    device = setup_environment(args)
    model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths(args)
    train_h5f, test_h5f, batch_num = load_datasets(args)
    train_idxs, val_idxs, test_idxs = generate_indices(batch_num, args.random_seed, test_h5f)
    model, optimizer, scheduler, params = initialize_model_and_optim(device, args.flanking_size)
    params["RANDOM_SEED"] = args.random_seed
    train_metric_files = create_metric_files(log_output_train_base)
    valid_metric_files = create_metric_files(log_output_val_base)
    test_metric_files = create_metric_files(log_output_test_base)
    train_model(model, optimizer, scheduler, train_h5f, test_h5f, train_idxs, val_idxs, test_idxs, 
                model_output_base, args, device, params, train_metric_files, valid_metric_files, test_metric_files)
    train_h5f.close()
    test_h5f.close()