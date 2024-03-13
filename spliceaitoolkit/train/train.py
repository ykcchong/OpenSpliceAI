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
                    --test_dataset="./data/test.h5" --disable_wandb=False
                    
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

import argparse
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import platform
from spliceaitoolkit.train.spliceai import *
from spliceaitoolkit.train.utils import *
from spliceaitoolkit.constants import *
import h5py
import time
import wandb # weights and biases: need to connect to this one

RANDOM_SEED = 42

def setup_device():
    """
    Selects the computational device (CUDA, MPS, or CPU) based on availability and platform.
    Checks for the availability of a CUDA-compatible GPU and uses it if available.
    On macOS (Darwin), it opts for Apple's Metal Performance Shaders (MPS) if available. Otherwise,
    it falls back to using the CPU.

    Returns:
    - torch.device: The selected computational device.
    """
    device_str = "cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" else "cpu"
    return torch.device(device_str)


def initialize_paths(output_dir, project_name, flanking_size, exp_num, sequence_length, model_arch, loss_fun):
    """
    Initializes and creates project directories for storing model outputs and logs.

    Parameters:
    - output_dir (str): Base directory for storing output files.
    - project_name (str): Name of the project, used in creating subdirectories.
    - flanking_size (int): Size of the flanking sequences used in the model.
    - exp_num (int): Experiment number for distinguishing between different runs.
    - sequence_length (int): Length of the sequences used in the model.
    - model_arch (str): Architecture of the model.
    - loss_fun (str): Loss function used for training the model.

    Returns:
    - str: path for model outputs
    - str: path for training logs
    - str: path for validation logs
    - str: path for test logs
    """
    ####################################
    # Modify the model verson here!!
    ####################################
    MODEL_VERSION = f"{model_arch}_{loss_fun}_{project_name}_{sequence_length}_{flanking_size}_{exp_num}"
    ####################################
    # Modify the model verson here!!
    ####################################
    model_train_outdir = f"{output_dir}/{MODEL_VERSION}/{exp_num}/"
    model_output_base = f"{model_train_outdir}models/"
    log_output_base = f"{model_train_outdir}LOG/"
    log_output_train_base = f"{log_output_base}TRAIN/"
    log_output_val_base = f"{log_output_base}VAL/"
    log_output_test_base = f"{log_output_base}TEST/"
    for path in [model_output_base, log_output_train_base, log_output_val_base, log_output_test_base]:
        os.makedirs(path, exist_ok=True)
    return model_output_base, log_output_train_base, log_output_val_base, log_output_test_base


def initialize_model_and_optim(device, flanking_size, model_arch):
    """
    Initializes the SpliceAI model, criterion (loss function), optimizer, and learning rate scheduler, 
    based on the provided flanking size and model architecture. 

    Parameters:
    - device (torch.device): The computational device to use (CUDA, MPS, or CPU).
    - flanking_size (int): The size of the flanking sequences, influencing the model architecture.
    - model_arch (str): The chosen architecture of the model, affecting how it's initialized.

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

    # criterion = nn.BCELoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, 1000, train_size * EPOCH_NUM)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 7, 8, 9], gamma=0.5)
    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE}

    return model, criterion, optimizer, scheduler, params

def load_data_from_shard(h5f, shard_idx, device, batch_size, params, shuffle=False):
    """
    Loads data from a specified shard of the dataset (in HDF5 format) and 
    prepares it as tensors wrapped into DataLoader for batch model training or evaluation.

    Parameters:
    - h5f (h5py.File): The HDF5 file object containing the dataset, divided into shards.
    - shard_idx (int): The index of the shard to load data from.
    - device (torch.device): The computational device (CUDA, MPS, CPU) the data should be prepared for.
    - batch_size (int): The number of samples to include in each batch.
    - params (dict): A dictionary of parameters, may include additional settings for data processing.
    - shuffle (bool, optional): Whether to shuffle the data in the DataLoader. Default is False.

    Returns:
    - DataLoader: A PyTorch DataLoader containing the dataset from the specified shard, ready for iteration.
    """

    X = h5f[f'X{shard_idx}'][:].transpose(0, 2, 1)
    Y = h5f[f'Y{shard_idx}'][0, ...].transpose(0, 2, 1)
    # print("\n\tX.shape: ", X.shape)
    # print("\tY.shape: ", Y.shape)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    ds = TensorDataset(X, Y)
    # print("\rds: ", ds)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, pin_memory=True)



def model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode, criterion):
    """
    Evaluates the model's performance on a batch of data and logs the metrics.
    Calculates various metrics, such as top-kL accuracy and AUPRC, for a given set of predictions and true labels.
    The results are written to specified log files and can also be logged to Weights & Biases if enabled.

    Parameters:
    - batch_ylabel (list of torch.Tensor): A list of tensors containing the true labels for each batch.
    - batch_ypred (list of torch.Tensor): A list of tensors containing the predicted labels for each batch.
    - metric_files (dict): A dictionary containing paths to files where metrics should be logged.
    - run_mode (str): The current phase of model usage ('train', 'validation', 'test') indicating where to log the metrics.
    - criterion (str): The loss function that was used during training or evaluation, for appropriate metric calculation.
    """

    batch_ylabel = torch.cat(batch_ylabel, dim=0)
    batch_ypred = torch.cat(batch_ypred, dim=0)
    is_expr = (batch_ylabel.sum(axis=(1,2)) >= 1).cpu().numpy()
    if np.any(is_expr):
        ############################
        # Topk SpliceAI assessment approach
        ############################
        subset_size = 1000
        indices = np.arange(batch_ylabel[is_expr].shape[0])
        subset_indices = np.random.choice(indices, size=min(subset_size, len(indices)), replace=False)
        Y_true_1 = batch_ylabel[is_expr][subset_indices, 1, :].flatten().cpu().detach().numpy()
        Y_true_2 = batch_ylabel[is_expr][subset_indices, 2, :].flatten().cpu().detach().numpy()
        Y_pred_1 = batch_ypred[is_expr][subset_indices, 1, :].flatten().cpu().detach().numpy()
        Y_pred_2 = batch_ypred[is_expr][subset_indices, 2, :].flatten().cpu().detach().numpy()
        acceptor_topkl_accuracy, acceptor_auprc = print_topl_statistics(np.asarray(Y_true_1),
                            np.asarray(Y_pred_1), metric_files["topk_acceptor"], type='acceptor', print_top_k=True)
        donor_topkl_accuracy, donor_auprc = print_topl_statistics(np.asarray(Y_true_2),
                            np.asarray(Y_pred_2), metric_files["topk_donor"], type='donor', print_top_k=True)
        if criterion == "cross_entropy_loss":
            loss = categorical_crossentropy_2d(batch_ylabel, batch_ypred)
        elif criterion == "focal_loss":
            loss = focal_loss(batch_ylabel, batch_ypred)
        for k, v in metric_files.items():
            with open(v, 'a') as f:
                if k == "loss_batch":
                    f.write(f"{loss.item()}\n")
                elif k == "topk_acceptor":
                    f.write(f"{acceptor_topkl_accuracy}\n")
                elif k == "topk_donor":
                    f.write(f"{donor_topkl_accuracy}\n")
                elif k == "auprc_acceptor":
                    f.write(f"{acceptor_auprc}\n")
                elif k == "auprc_donor":
                    f.write(f"{donor_auprc}\n")
        wandb.log({
            f'{run_mode}/loss_batch': loss.item(),
            f'{run_mode}/topk_acceptor': acceptor_topkl_accuracy,
            f'{run_mode}/topk_donor': donor_topkl_accuracy,
            f'{run_mode}/auprc_acceptor': acceptor_auprc,
            f'{run_mode}/auprc_donor': donor_auprc,
        })
        print("***************************************\n\n")
    batch_ylabel = []
    batch_ypred = []


def valid_epoch(model, h5f, idxs, batch_size, criterion, device, params, metric_files, run_mode, sample_freq):
    """
    Validates the SpliceAI model on a given dataset.
    (Similar to train_epoch, but without performing backpropagation or updating model parameters)

    Parameters:
    - model (torch.nn.Module): The SpliceAI model to be evaluated.
    - h5f (h5py.File): HDF5 file object containing the validation or test data.
    - idxs (np.array): Array of indices for the batches to be used in validation/testing.
    - batch_size (int): Size of each batch.
    - criterion (str): The loss function used for validation/testing.
    - device (torch.device): The computational device (CUDA, MPS, CPU).
    - params (dict): Dictionary of parameters related to model and validation/testing.
    - metric_files (dict): Dictionary containing paths to log files for various metrics.
    - run_mode (str): Indicates the phase (e.g., "validation", "test").
    - sample_freq (int): Frequency of sampling for evaluation and logging.
    """

    print(f"\033[1m{run_mode.capitalize()}ing model...\033[0m")
    model.eval()
    running_loss = 0.0
    np.random.seed(RANDOM_SEED)  # You can choose any number as a seed
    shuffled_idxs = np.random.choice(idxs, size=len(idxs), replace=False)    
    print("shuffled_idxs: ", shuffled_idxs)
    batch_ylabel = []
    batch_ypred = []
    print_dict = {}
    batch_idx = 0
    for i, shard_idx in enumerate(shuffled_idxs, 1):
        print(f"Shard {i}/{len(shuffled_idxs)}")
        loader = load_data_from_shard(h5f, shard_idx, device, batch_size, params, shuffle=False)
        pbar = tqdm(loader, leave=False, total=len(loader), desc=f'Shard {i}/{len(shuffled_idxs)}')
        for batch in pbar:
            DNAs, labels = batch[0].to(device), batch[1].to(device)
            # print("\n\tDNAs.shape: ", DNAs.shape)
            # print("\tlabels.shape: ", labels.shape)
            DNAs, labels = clip_datapoints(DNAs, labels, params["CL"], 2)
            DNAs, labels = DNAs.to(torch.float32).to(device), labels.to(torch.float32).to(device)
            # print("\n\tAfter clipping DNAs.shape: ", DNAs.shape)
            # print("\tAfter clipping labels.shape: ", labels.shape)
            yp = model(DNAs)
            if criterion == "cross_entropy_loss":
                loss = categorical_crossentropy_2d(labels, yp)
            elif criterion == "focal_loss":
                loss = focal_loss(labels, yp)
            # Logging loss for every update.
            with open(metric_files["loss_every_update"], 'a') as f:
                f.write(f"{loss.item()}\n")
            wandb.log({
                f'{run_mode}/loss_every_update': loss.item(),
            })
            running_loss += loss.item()
            # print("loss: ", loss.item())
            batch_ylabel.append(labels.detach().cpu())
            batch_ypred.append(yp.detach().cpu())
            print_dict["loss"] = loss.item()
            pbar.set_postfix(print_dict)
            pbar.update(1)
            batch_idx += 1
            # if batch_idx % sample_freq == 0:
            #     model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode)
        pbar.close()
    model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode, criterion)


def train_epoch(model, h5f, idxs, batch_size, criterion, optimizer, scheduler, device, params, metric_files, run_mode, sample_freq):
    """
    Performs one epoch of training on the SpliceAI model.

    This function iterates over the dataset, performs the forward pass, calculates the loss,
    performs the backward pass, and updates the model parameters. It also logs various metrics.

    Parameters:
    - model (torch.nn.Module): The SpliceAI model to be trained.
    - h5f (h5py.File): HDF5 file object containing the training data.
    - idxs (np.array): Array of indices for the training batches.
    - batch_size (int): Size of each batch.
    - criterion (str): The loss function used for training.
    - optimizer (torch.optim.Optimizer): Optimizer used for training.
    - scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
    - device (torch.device): The computational device (CUDA, MPS, CPU).
    - params (dict): Dictionary of parameters related to model and training.
    - metric_files (dict): Dictionary containing paths to log files for various metrics.
    - run_mode (str): Indicates the phase of training (e.g., "train", "validation").
    - sample_freq (int): Frequency of sampling for evaluation and logging.
    """

    print(f"\033[1m{run_mode.capitalize()}ing model...\033[0m")
    model.train()
    running_loss = 0.0
    np.random.seed(RANDOM_SEED)  # You can choose any number as a seed
    shuffled_idxs = np.random.choice(idxs, size=len(idxs), replace=False)
    print("shuffled_idxs: ", shuffled_idxs)
    batch_ylabel = []
    batch_ypred = []
    print_dict = {}
    batch_idx = 0
    for i, shard_idx in enumerate(shuffled_idxs, 1):
        print(f"Shard {i}/{len(shuffled_idxs)}")
        loader = load_data_from_shard(h5f, shard_idx, device, batch_size, params, shuffle=True)
        pbar = tqdm(loader, leave=False, total=len(loader), desc=f'Shard {i}/{len(shuffled_idxs)}')
        for batch in pbar:
            DNAs, labels = batch[0].to(device), batch[1].to(device)
            # print("\n\tDNAs.shape: ", DNAs.shape)
            # print("\tlabels.shape: ", labels.shape)
            DNAs, labels = clip_datapoints(DNAs, labels, params["CL"], 2)
            DNAs, labels = DNAs.to(torch.float32).to(device), labels.to(torch.float32).to(device)
            # print("\n\tAfter clipping DNAs.shape: ", DNAs.shape)
            # print("\tAfter clipping labels.shape: ", labels.shape)
            optimizer.zero_grad()
            yp = model(DNAs)
            if criterion == "cross_entropy_loss":
                loss = categorical_crossentropy_2d(labels, yp)
            elif criterion == "focal_loss":
                loss = focal_loss(labels, yp)
            # Logging loss for every update.
            with open(metric_files["loss_every_update"], 'a') as f:
                f.write(f"{loss.item()}\n")
            wandb.log({
                f'{run_mode}/loss_every_update': loss.item(),
            })
            # print("loss: ", loss.item())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_ylabel.append(labels.detach().cpu())
            batch_ypred.append(yp.detach().cpu())
            print_dict["loss"] = loss.item()
            pbar.set_postfix(print_dict)
            pbar.update(1)
            batch_idx += 1
            # if batch_idx % sample_freq == 0:
            #     model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode)
        pbar.close()
    model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode, criterion)


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
        - disable_wandb (bool): Flag to disable logging to Weights & Biases.
    """

    output_dir = args.output_dir
    project_name = args.project_name
    sequence_length = 5000
    flanking_size = int(args.flanking_size)
    exp_num = args.exp_num
    model_arch = args.model
    assert int(flanking_size) in [80, 400, 2000, 10000]
    # assert training_target in ["RefSeq", "MANE", "SpliceAI", "SpliceAI27"]
    if args.disable_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    wandb.init(project=f'{project_name}', reinit=True)
    device = setup_device()
    print("device: ", device, file=sys.stderr)
    model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths(output_dir, project_name, flanking_size, exp_num, sequence_length, model_arch, args.loss)
    print("* Project name: ", args.project_name, file=sys.stderr)
    print("* Model_output_base: ", model_output_base, file=sys.stderr)
    print("* Log_output_train_base: ", log_output_train_base, file=sys.stderr)
    print("* Log_output_val_base: ", log_output_val_base, file=sys.stderr)
    print("* Log_output_test_base: ", log_output_test_base, file=sys.stderr)
    training_dataset = args.train_dataset
    testing_dataset = args.test_dataset
    print("Training_dataset: ", training_dataset, file=sys.stderr)
    print("Testing_dataset: ", testing_dataset, file=sys.stderr)
    print("Model architecture: ", model_arch, file=sys.stderr)
    print("Loss function: ", args.loss, file=sys.stderr)
    print("Flanking sequence size: ", args.flanking_size, file=sys.stderr)
    print("Exp number: ", args.exp_num, file=sys.stderr)
    train_h5f = h5py.File(training_dataset, 'r')
    test_h5f = h5py.File(testing_dataset, 'r')
    batch_num = len(train_h5f.keys()) // 2
    print("Batch_num: ", batch_num, file=sys.stderr)
    np.random.seed(RANDOM_SEED)  # You can choose any number as a seed
    idxs = np.random.permutation(batch_num)
    train_idxs = idxs[:int(0.9 * batch_num)]
    val_idxs = idxs[int(0.9 * batch_num):]
    test_idxs = np.arange(len(test_h5f.keys()) // 2)
    # train_idxs = idxs[:int(0.1*batch_num)]
    # val_idxs = idxs[int(0.2*batch_num):int(0.25*batch_num)]
    # test_idxs = np.arange(len(test_h5f.keys()) // 10)
    print("train_idxs: ", train_idxs, file=sys.stderr)
    print("val_idxs: ", val_idxs, file=sys.stderr)
    print("test_idxs: ", test_idxs, file=sys.stderr)
    model, criterion, optimizer, scheduler, params = initialize_model_and_optim(device, flanking_size, model_arch)
    train_metric_files = {
        'topk_donor': f'{log_output_train_base}/donor_topk.txt',
        'auprc_donor': f'{log_output_train_base}/donor_accuracy.txt',
        'topk_acceptor': f'{log_output_train_base}/acceptor_topk.txt',
        'auprc_acceptor': f'{log_output_train_base}/acceptor_accuracy.txt',
        'loss_batch': f'{log_output_train_base}/loss_batch.txt',
        'loss_every_update': f'{log_output_train_base}/loss_every_update.txt'
    }
    valid_metric_files = {
        'topk_donor': f'{log_output_val_base}/donor_topk.txt',
        'auprc_donor': f'{log_output_val_base}/donor_accuracy.txt',
        'topk_acceptor': f'{log_output_val_base}/acceptor_topk.txt',
        'auprc_acceptor': f'{log_output_val_base}/acceptor_accuracy.txt',
        'loss_batch': f'{log_output_val_base}/loss_batch.txt',
        'loss_every_update': f'{log_output_val_base}/loss_every_update.txt'
    }
    test_metric_files = {
        'topk_donor': f'{log_output_test_base}/donor_topk.txt',
        'auprc_donor': f'{log_output_test_base}/donor_accuracy.txt',
        'topk_acceptor': f'{log_output_test_base}/acceptor_topk.txt',
        'auprc_acceptor': f'{log_output_test_base}/acceptor_accuracy.txt',
        'loss_batch': f'{log_output_test_base}/loss_batch.txt',
        'loss_every_update': f'{log_output_test_base}/loss_every_update.txt'
    }
    SAMPLE_FREQ = 1000
    for epoch in range(EPOCH_NUM):
        print("\n--------------------------------------------------------------")
        print(f">> Epoch {epoch + 1}")
        start_time = time.time()
        train_epoch(model, train_h5f, train_idxs, params["BATCH_SIZE"], args.loss, optimizer, scheduler, device, params, train_metric_files, run_mode="train", sample_freq=SAMPLE_FREQ)
        valid_epoch(model, train_h5f, val_idxs, params["BATCH_SIZE"], args.loss, device, params, valid_metric_files, run_mode="validation", sample_freq=SAMPLE_FREQ)
        valid_epoch(model, test_h5f, test_idxs, params["BATCH_SIZE"], args.loss, device, params, test_metric_files, run_mode="test", sample_freq=SAMPLE_FREQ)
        torch.save(model.state_dict(), f"{model_output_base}/model_{epoch}.pt")
        print("--- %s seconds ---" % (time.time() - start_time))
        print("--------------------------------------------------------------")
    train_h5f.close()
    test_h5f.close()