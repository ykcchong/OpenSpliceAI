import logging
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler, random_split
from tqdm import tqdm
from Bio import SeqIO
import platform
from pathlib import Path
import splan
from splan_utils import *
from splan_constant import *
from tqdm import tqdm
import h5py
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import wandb

from torch.utils.data import Dataset
from utils import clip_datapoints, print_topl_statistics


def setup_device():
    """Select computation device based on availability."""
    device_str = "cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" else "cpu"
    return torch.device(device_str)


def top_k_accuracy(pred_probs, labels):
    pred_probs, labels = map(lambda x: x.view(-1), [pred_probs, labels])  # Flatten
    k = (labels == 1.0).sum().item()
    _, top_k_indices = pred_probs.topk(k)
    correct = labels[top_k_indices] == 1.0
    return correct.float().mean()


def initialize_paths(chunk_size, flanking_size, exp_num, target):
    """Initialize project directories and create them if they don't exist."""
    MODEL_VERSION = f"{target}_splan_{chunk_size}chunk_{flanking_size}flank_spliceai_architecture"
    project_root = "/Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/"
    data_dir = f"{project_root}results/{target}/"
    model_train_outdir = f"{project_root}results/model_train_outdir/{MODEL_VERSION}/{exp_num}/"
    model_output_base = f"{model_train_outdir}models/"
    log_output_base = f"{model_train_outdir}LOG/"
    log_output_train_base = f"{log_output_base}TRAIN/"
    log_output_val_base = f"{log_output_base}VAL/"
    log_output_test_base = f"{log_output_base}TEST/"
    for path in [model_output_base, log_output_train_base, log_output_val_base, log_output_test_base]:
        os.makedirs(path, exist_ok=True)
    return data_dir, model_output_base, log_output_train_base, log_output_val_base, log_output_test_base


def initialize_model_and_optim(device, flanking_size):
    """Initialize the model, criterion, optimizer, and scheduler."""
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
    model = splan.SPLAN(L, W, AR, int(flanking_size)).to(device)
    # criterion = nn.BCELoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, 1000, train_size * EPOCH_NUM)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 7, 8, 9], gamma=0.5)
    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE}
    return model, criterion, optimizer, scheduler, params


def calculate_metrics(y_true, y_pred):
    """Calculate metrics including precision, recall, f1-score, and accuracy."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy


def threshold_predictions(y_probs, threshold=0.5):
    """Threshold probabilities to get binary predictions."""
    return (y_probs > threshold).astype(int)


def load_data_from_shard(h5f, shard_idx, device, batch_size, params, shuffle):
    X = h5f[f'X{shard_idx}'][:].transpose(0, 2, 1)
    Y = h5f[f'Y{shard_idx}'][0, ...]
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=8, pin_memory=True)


def process_batch(batch, device, params, model, criterion, optimizer=None, run_mode="train"):
    DNAs, labels = batch[0].to(device), batch[1].to(device)
    DNAs, labels = clip_datapoints(DNAs, labels, params["CL"], 2)
    DNAs, labels = DNAs.to(torch.float32).to(device), labels.to(torch.float32).to(device)
    if run_mode =="train":
        optimizer.zero_grad()
    loss, yp = model_fn(DNAs, labels, model, criterion)
    if run_mode =="train":
        loss.backward()
        optimizer.step()
    return loss.item(), yp, labels


def run_epoch(model, h5f, idxs, batch_size, criterion, optimizer, scheduler, device, params, metric_files, run_mode):
    model.train() if run_mode == "train" else model.eval()
    running_loss = 0.0
    shuffled_idxs = np.random.choice(idxs, size=len(idxs), replace=False)    
    batch_idx = 1
    batch_ylabel = []
    batch_ypred = []
    step_size = 200
    for i, shard_idx in enumerate(shuffled_idxs, 1):
        loader = load_data_from_shard(h5f, shard_idx, device, batch_size, params, shuffle=run_mode)
        pbar = tqdm(loader, leave=False, total=len(loader), desc=f'Shard {i}/{len(shuffled_idxs)}')
        for batch in pbar:
            loss, yp, labels = process_batch(batch, device, params, model, criterion, optimizer, run_mode)
            running_loss += loss
            batch_ylabel.append(labels.detach().cpu())
            batch_ypred.append(yp.detach().cpu())
            print_dict = {}
            print_dict["loss"] = loss
            pbar.set_postfix(print_dict)
            pbar.update(1)
            # Each step is 20 updates
            if batch_idx % step_size == 0:
                batch_ylabel = torch.cat(batch_ylabel, dim=0)
                batch_ypred = torch.cat(batch_ypred, dim=0)
                is_expr = (batch_ylabel.sum(axis=(1,2)) >= 1).cpu().numpy()
                if np.any(is_expr):
                    ############################
                    # My assessment
                    ############################
                    print("\n***************************************")
                    print(f"\033[1m{run_mode} set metrics:\033[0m")
                    loss = categorical_crossentropy_2d(batch_ylabel, batch_ypred)
                    print("\tloss: ", loss.item())
                    with open(metric_files["loss"], 'a') as f:
                        f.write(f"{loss.item()}\n")
                    metrics = {}
                    for role, idx in [("neither", 0), ("donor", 1), ("acceptor", 2)]:
                        print(f"\n\t\033[1m{role} metrics:\033[0m\n\t", end="")
                        y_true = batch_ylabel[is_expr, :, idx].flatten().cpu().detach().numpy()
                        y_pred = threshold_predictions(batch_ypred[is_expr, :, idx].flatten().cpu().detach().numpy())
                        metrics[f"{role}_precision"], metrics[f"{role}_recall"], metrics[f"{role}_f1"], metrics[f"{role}_accuracy"] = calculate_metrics(y_true, y_pred)
                        for name, value in metrics.items():
                            if metric_files and name in metric_files:
                                with open(metric_files[name], 'a') as f:
                                    f.write(f"{value}\n")
                            print(f"{name}: {value:.4f}\t", end="")
                        metrics = {}
                    ############################
                    # Topk SpliceAI assessment approach
                    ############################
                    Y_true_1 = batch_ylabel[is_expr, :, 1].flatten().cpu().detach().numpy()
                    Y_true_2 = batch_ylabel[is_expr, :, 2].flatten().cpu().detach().numpy()
                    Y_pred_1 = batch_ypred[is_expr, :, 1].flatten().cpu().detach().numpy()
                    Y_pred_2 = batch_ypred[is_expr, :, 2].flatten().cpu().detach().numpy()
                    print_topl_statistics(np.asarray(Y_true_1),
                                        np.asarray(Y_pred_1), metric_files["acceptor_topk"], type='acceptor', print_top_k=True)
                    print_topl_statistics(np.asarray(Y_true_2),
                                        np.asarray(Y_pred_2), metric_files["donor_topk"], type='donor', print_top_k=True)
                    print("***************************************\n\n")
                batch_ylabel, batch_ypred = [], []                
            batch_idx += 1
        pbar.close()
    if run_mode == "train":
        scheduler.step()


def main():
    # os.environ['WANDB_MODE'] = 'disabled'
    # wandb.init(project='spliceai-general', reinit=True)
    chunk_size = sys.argv[1]
    flanking_size = sys.argv[2]
    exp_num = sys.argv[3]
    training_target = sys.argv[4]
    assert int(flanking_size) in [80, 400, 2000, 10000]
    assert training_target in ["MANE", "SpliceAI"]
    device = setup_device()
    # target = "train_test_dataset_SpliceAI"
    target = f"train_test_dataset_{training_target}"
    data_dir, model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths(chunk_size, flanking_size, exp_num, target)
    print("* data_dir: ", data_dir)
    print("* model_output_base: ", model_output_base)
    print("* log_output_train_base: ", log_output_train_base)
    print("* log_output_val_base: ", log_output_val_base)
    print("* log_output_test_base: ", log_output_test_base)
    training_dataset = f"{data_dir}dataset_train.h5"
    testing_dataset = f"{data_dir}dataset_test.h5"
    train_h5f = h5py.File(training_dataset, 'r')
    test_h5f = h5py.File(testing_dataset, 'r')
    batch_num = len(train_h5f.keys()) // 2
    print("batch_num: ", batch_num)
    idxs = np.random.permutation(batch_num)
    train_idxs = idxs[:int(0.9 * batch_num)]
    val_idxs = idxs[int(0.9 * batch_num):]
    # test_idxs = np.arange(2)
    # train_idxs = idxs[:int(0.9 * batch_num)]
    # val_idxs = idxs[int(0.9 * batch_num):]
    test_idxs = np.arange(len(test_h5f.keys()) // 2)
    model, criterion, optimizer, scheduler, params = initialize_model_and_optim(device, flanking_size)
    train_metric_files = {
        'neither_precision': f'{log_output_train_base}/neither_precision.txt',
        'neither_recall': f'{log_output_train_base}/neither_recall.txt',
        'neither_f1': f'{log_output_train_base}/neither_f1.txt',
        'neither_accuracy': f'{log_output_train_base}/neither_accuracy.txt',
        'donor_topk': f'{log_output_train_base}/donor_topk.txt',
        'donor_precision': f'{log_output_train_base}/donor_precision.txt',
        'donor_recall': f'{log_output_train_base}/donor_recall.txt',
        'donor_f1': f'{log_output_train_base}/donor_f1.txt',
        'donor_accuracy': f'{log_output_train_base}/donor_accuracy.txt',
        'acceptor_topk': f'{log_output_train_base}/acceptor_topk.txt',
        'acceptor_precision': f'{log_output_train_base}/acceptor_precision.txt',
        'acceptor_recall': f'{log_output_train_base}/acceptor_recall.txt',
        'acceptor_f1': f'{log_output_train_base}/acceptor_f1.txt',
        'acceptor_accuracy': f'{log_output_train_base}/acceptor_accuracy.txt',
        'loss': f'{log_output_train_base}/loss.txt'
    }
    valid_metric_files = {
        'neither_precision': f'{log_output_val_base}/neither_precision.txt',
        'neither_recall': f'{log_output_val_base}/neither_recall.txt',
        'neither_f1': f'{log_output_val_base}/neither_f1.txt',
        'neither_accuracy': f'{log_output_val_base}/neither_accuracy.txt',
        'donor_topk': f'{log_output_val_base}/donor_topk.txt',
        'donor_precision': f'{log_output_val_base}/donor_precision.txt',
        'donor_recall': f'{log_output_val_base}/donor_recall.txt',
        'donor_f1': f'{log_output_val_base}/donor_f1.txt',
        'donor_accuracy': f'{log_output_val_base}/donor_accuracy.txt',
        'acceptor_topk': f'{log_output_val_base}/acceptor_topk.txt',
        'acceptor_precision': f'{log_output_val_base}/acceptor_precision.txt',
        'acceptor_recall': f'{log_output_val_base}/acceptor_recall.txt',
        'acceptor_f1': f'{log_output_val_base}/acceptor_f1.txt',
        'acceptor_accuracy': f'{log_output_val_base}/acceptor_accuracy.txt',
        'loss': f'{log_output_val_base}/loss.txt'
    }
    test_metric_files = {
        'neither_precision': f'{log_output_test_base}/neither_precision.txt',
        'neither_recall': f'{log_output_test_base}/neither_recall.txt',
        'neither_f1': f'{log_output_test_base}/neither_f1.txt',
        'neither_accuracy': f'{log_output_test_base}/neither_accuracy.txt',
        'donor_topk': f'{log_output_test_base}/donor_topk.txt',
        'donor_precision': f'{log_output_test_base}/donor_precision.txt',
        'donor_recall': f'{log_output_test_base}/donor_recall.txt',
        'donor_f1': f'{log_output_test_base}/donor_f1.txt',
        'donor_accuracy': f'{log_output_test_base}/donor_accuracy.txt',
        'acceptor_topk': f'{log_output_test_base}/acceptor_topk.txt',
        'acceptor_precision': f'{log_output_test_base}/acceptor_precision.txt',
        'acceptor_recall': f'{log_output_test_base}/acceptor_recall.txt',
        'acceptor_f1': f'{log_output_test_base}/acceptor_f1.txt',
        'acceptor_accuracy': f'{log_output_test_base}/acceptor_accuracy.txt',
        'loss': f'{log_output_test_base}/loss.txt'
    }
    for epoch in range(EPOCH_NUM):
        print("\n--------------------------------------------------------------")
        print(f">> Epoch {epoch + 1}")
        start_time = time.time()
        # train(model, train_h5f, train_idxs, BATCH_SIZE, criterion, optimizer, scheduler, device, params, train_metric_files)
        # validate_test_model(model, train_h5f, val_idxs, BATCH_SIZE, criterion, device, params, valid_metric_files, "validation")
        # validate_test_model(model, test_h5f, test_idxs, BATCH_SIZE, criterion, device, params, test_metric_files, "test")
        run_epoch(model, train_h5f, train_idxs, params["BATCH_SIZE"], criterion, optimizer, scheduler, device, params, train_metric_files, run_mode="train")
        run_epoch(model, train_h5f, val_idxs, params["BATCH_SIZE"], criterion, None, None, device, params, valid_metric_files, run_mode="validation")
        run_epoch(model, test_h5f, test_idxs, params["BATCH_SIZE"], criterion, None, None, device, params, test_metric_files, run_mode="test")
        torch.save(model.state_dict(), f"{model_output_base}/model_{epoch}.pt")
        print("--- %s seconds ---" % (time.time() - start_time))
        print("--------------------------------------------------------------")
    train_h5f.close()
    test_h5f.close()

if __name__ == "__main__":
    main()
