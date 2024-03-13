import argparse
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import platform
import h5py
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from spliceaitoolkit.predict.spliceai import *
from spliceaitoolkit.predict.utils import *
from spliceaitoolkit.constants import *
import wandb

RANDOM_SEED = 42

# needed
def setup_device():
    """Select computation device based on availability."""
    device_str = "cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" else "cpu"
    return torch.device(device_str)

# adjusted for prediction only
def initialize_paths(output_dir, flanking_size, sequence_length, model_arch):
    """Initialize project directories and create them if they don't exist."""

    BASENAME = f"{model_arch}_{sequence_length}_{flanking_size}"
    model_train_outdir = f"{output_dir}/{BASENAME}/"

    log_output_base = f"{model_train_outdir}LOG/"
    os.makedirs(log_output_base, exist_ok=True)

    return log_output_base

# adjusted for prediction 
def load_model(device, flanking_size, model_arch):
    """Loads the given model."""
    # Hyper-parameters:
    # L: Number of convolution kernels
    # W: Convolution window size in each residual unit
    # AR: Atrous rate in each residual unit
    L = 32
    W = np.asarray([11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1])
    N_GPUS = 2
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
    
    # GET MODEL
    model = SpliceAI(L, W, AR).to(device)
    print(model, file=sys.stderr)
    return model


# don't need?
def load_data_from_shard(h5f, shard_idx, device, batch_size, params, shuffle=False):
    X = h5f[f'X{shard_idx}'][:].transpose(0, 2, 1)
    Y = h5f[f'Y{shard_idx}'][0, ...].transpose(0, 2, 1)
    # print("\n\tX.shape: ", X.shape)
    # print("\tY.shape: ", Y.shape)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    ds = TensorDataset(X, Y)

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, pin_memory=True)

# definitely need
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


def predict(args):
    print("Running SpliceAI-toolkit with 'predict' mode")

    # one-hot encode input sequence -> to DataLoader -> model.eval() -> get predictions -> return
    
    # get args
    output_dir = args.output_dir
    sequence_length = SL
    flanking_size = int(args.flanking_size)
    model_arch = args.model

    assert int(flanking_size) in [80, 400, 2000, 10000]

    # get device for model
    device = setup_device()
    print("device: ", device, file=sys.stderr)

    # initialize output directory
    log_output_base = initialize_paths(output_dir, flanking_size, sequence_length, model_arch)
    print("* Project name: ", args.project_name, file=sys.stderr)
    print("* log_output_base: ", log_output_base, file=sys.stderr)

    print("Model architecture: ", model_arch, file=sys.stderr)
    print("Loss function: ", args.loss, file=sys.stderr)
    print("Flanking sequence size: ", args.flanking_size, file=sys.stderr)

    train_h5f = h5py.File(training_dataset, 'r')
    test_h5f = h5py.File(testing_dataset, 'r')

    batch_num = len(train_h5f.keys()) // 2
    print("Batch_num: ", batch_num, file=sys.stderr)
    np.random.seed(RANDOM_SEED)  # You can choose any number as a seed
    idxs = np.random.permutation(batch_num)
    train_idxs = idxs[:int(0.9 * batch_num)]
    val_idxs = idxs[int(0.9 * batch_num):]
    test_idxs = np.arange(len(test_h5f.keys()) // 2)
    print("train_idxs: ", train_idxs, file=sys.stderr)
    print("val_idxs: ", val_idxs, file=sys.stderr)
    print("test_idxs: ", test_idxs, file=sys.stderr)

    model = load_model(device, flanking_size, model_arch)

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