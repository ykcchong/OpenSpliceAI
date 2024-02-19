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
# import gene_dataset_chunk
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


def initialize_paths(chunk_size, flanking_size, exp_num):
    """Initialize project directories and create them if they don't exist."""
    MODEL_VERSION = f"splan_{chunk_size}chunk_{flanking_size}flank_spliceai_architecture"
    project_root = "/Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/"
    data_dir = f"{project_root}results/train_test_dataset_SpliceAI/"
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
    # # Hyper-parameters:
    # # L: Number of convolution kernels
    # # W: Convolution window size in each residual unit
    # # AR: Atrous rate in each residual unit
    L = 32
    N_GPUS = 2
    W = np.asarray([11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1])
    # BATCH_SIZE = 18*N_GPUS

    if int(flanking_size) == 80:
        W = np.asarray([11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1])
        # BATCH_SIZE = 18*N_GPUS
    elif int(flanking_size) == 400:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
        # BATCH_SIZE = 18*N_GPUS
    elif int(flanking_size) == 2000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10])
        # BATCH_SIZE = 12*N_GPUS
    elif int(flanking_size) == 10000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21, 41, 41, 41, 41])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10, 25, 25, 25, 25])
        # BATCH_SIZE = 6*N_GPUS

    CL = 2 * np.sum(AR*(W-1))
    # assert CL <= CL_max and CL == int(sys.argv[1])
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


def train(model, h5f, train_idxs, batch_size, criterion, optimizer, scheduler, device, params, metric_files):
    print("Training the model...")
    model.train()
    # running_output, running_label = [], []
    epoch_loss = 0    
    train_shuffled_idxs = np.random.choice(train_idxs, size=len(train_idxs), replace=False)
    print("train_shuffled_idxs: ", train_shuffled_idxs)
    batch_idx = 0
    for i, shard_idx in enumerate(train_shuffled_idxs, 1):
        X = h5f[f'X{shard_idx}'][:].transpose(0, 2, 1)
        Y = h5f[f'Y{shard_idx}'][0, ...]
        # print("X.shape: ", X.shape)
        # print("Y.shape: ", Y.shape)
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)  # TODO: Check whether drop_last=True?

        pbar = tqdm(loader, leave=False, total=len(loader), desc=f'Shard {i}/{len(train_idxs)}')
                    # , desc=f"Train Epoch", unit="batch")
        # print("pbar: ", len(pbar))
        for batch in pbar:
            # X, Y = batch[0].cuda(), batch[1].cuda()
            DNAs, labels = batch[0].to(device), batch[1].to(device)
            # print("Org DNAs.shape: ", DNAs.shape)
            # print("Org labels.shape: ", labels.shape)
            DNAs, labels = clip_datapoints(DNAs, labels, params["CL"], 2) 
            # print("After clipping DNAs.shape: ", DNAs.shape)
            # print("After clipping labels.shape: ", labels[0].shape)
            DNAs, labels = DNAs.to(torch.float32).to(device), labels.to(torch.float32).to(device)
            # print("Rotate DNAs.shape: ", DNAs.shape)
            # print("Rotate labels.shape: ", labels.shape)

            optimizer.zero_grad()
            loss, yp = model_fn(DNAs, labels, model, criterion)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # running_output.append(yp.detach().cpu())
            # running_label.append(labels.detach().cpu())

            is_expr = (labels.sum(axis=(1,2)) >= 1).cpu().numpy()
            if np.any(is_expr):  # Process metrics only if there are expressions
                ############################
                # My assessment
                ############################
                metrics = {}
                metrics[f"loss"] = loss.item() 
                for role, idx in [("neither", 0), ("donor", 1), ("acceptor", 2)]:
                    # print("role: ", role, "\tidx: ", idx)
                    # print("labels: ", labels.shape)
                    # print("yp: ", yp.shape)
                    y_true = labels[is_expr, :, idx].flatten().cpu().detach().numpy()
                    y_pred = threshold_predictions(yp[is_expr, :, idx].flatten().cpu().detach().numpy())
                    # print("y_true: ", y_true)
                    # print("y_pred: ", y_pred)
                    metrics[f"{role}_precision"], metrics[f"{role}_recall"], metrics[f"{role}_f1"], metrics[f"{role}_accuracy"] = calculate_metrics(y_true, y_pred)
                    # Write metrics to files
                    for name, value in metrics.items():
                        if metric_files and name in metric_files:
                            with open(metric_files[name], 'a') as f:
                                f.write(f"{value}\n")
                # # Update progress bar
                # # pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})
                # print_dict = {}
                # for k, v in metrics.items():
                #     if k not in ['donor_accuracy', 'acceptor_accuracy', 'neither_accuracy', 'neither_precision', 'neither_recall', 'neither_f1']:
                #         print_dict[k] = f"{v:.4f}"
                # pbar.set_postfix(print_dict)

                ############################
                # Topk SpliceAI assessment approach
                ############################
                print_top_k = False
                if batch_idx % 100 == 0:
                    print_top_k = True
                if print_top_k:
                    print("\n\033[1mTraining set metrics:\033[0m")
                # print("labels: ", labels.shape)
                # print("yp: ", yp.shape)
                Y_true_1 = labels[is_expr, :, 1].flatten().cpu().detach().numpy()
                Y_true_2 = labels[is_expr, :, 2].flatten().cpu().detach().numpy()
                Y_pred_1 = yp[is_expr, :, 1].flatten().cpu().detach().numpy()
                Y_pred_2 = yp[is_expr, :, 2].flatten().cpu().detach().numpy()
                acceptor_top1, acceptor_auprc = print_topl_statistics(np.asarray(Y_true_1),
                                      np.asarray(Y_pred_1), metric_files["acceptor_topk"], type='acceptor', print_top_k=print_top_k)
                donor_top1, donor_auprc = print_topl_statistics(np.asarray(Y_true_2),
                                    np.asarray(Y_pred_2), metric_files["donor_topk"], type='donor', print_top_k=print_top_k)
                
                # Update progress bar
                print_dict = {}
                print_dict["loss"] = metrics["loss"]
                print_dict["acceptor_top1"] = f"{acceptor_top1:.4f}"
                print_dict["acceptor_auprc"] = f"{acceptor_auprc:.4f}"
                print_dict["acceptor_precision"] = f"{metrics['acceptor_precision']:.4f}"
                print_dict["donor_top1"] = f"{donor_top1:.4f}"
                print_dict["donor_auprc"] = f"{donor_auprc:.4f}"
                print_dict["donor_precision"] = f"{metrics['donor_precision']:.4f}"
                pbar.set_postfix(print_dict)

                if print_top_k:
                    print("\n\n")
            pbar.update(1)
            batch_idx += 1
        pbar.close()


def validate_test_model(model, h5f, idxs, batch_size, criterion, device, params, metric_files, run_mode):
    if run_mode == "validation":
        print("Validating the model...")
    elif run_mode == "test":
        print("Testing the model...")
    model.eval()
    epoch_loss = 0
    val_shuffled_idxs = np.random.choice(idxs, size=len(idxs), replace=False)
    batch_idx = 0
    for i, shard_idx in enumerate(val_shuffled_idxs, 1):
        X = h5f[f'X{shard_idx}'][:].transpose(0, 2, 1)
        Y = h5f[f'Y{shard_idx}'][0, ...]
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8, pin_memory=True)  # TODO: Check whether drop_last=True?
        pbar = tqdm(loader, leave=False, total=len(loader), desc=f'Shard {i}/{len(val_shuffled_idxs)}')
        for batch in pbar:
            DNAs, labels = batch[0].to(device), batch[1].to(device)
            DNAs, labels = clip_datapoints(DNAs, labels, params["CL"], 2)
            DNAs, labels = DNAs.to(torch.float32).to(device), labels.to(torch.float32).to(device)
            loss, yp = model_fn(DNAs, labels, model, criterion)
            epoch_loss += loss.item()

            is_expr = (labels.sum(axis=(1,2)) >= 1).cpu().numpy()
            if np.any(is_expr):  # Process metrics only if there are expressions
                ############################
                # My assessment
                ############################
                metrics = {}
                metrics[f"loss"] = loss.item() 
                for role, idx in [("neither", 0), ("donor", 1), ("acceptor", 2)]:
                    # print("role: ", role, "\tidx: ", idx)
                    # print("labels: ", labels.shape)
                    # print("yp: ", yp.shape)
                    y_true = labels[is_expr, :, idx].flatten().cpu().detach().numpy()
                    y_pred = threshold_predictions(yp[is_expr, :, idx].flatten().cpu().detach().numpy())
                    # print("y_true: ", len(y_true))
                    # print("y_pred: ", len(y_pred))
                    metrics[f"{role}_precision"], metrics[f"{role}_recall"], metrics[f"{role}_f1"], metrics[f"{role}_accuracy"] = calculate_metrics(y_true, y_pred)
                    # Write metrics to files
                    for name, value in metrics.items():
                        if metric_files and name in metric_files:
                            with open(metric_files[name], 'a') as f:
                                f.write(f"{value}\n")
                # # Update progress bar
                # print_dict = {}
                # for k, v in metrics.items():
                #     if k not in ['donor_accuracy', 'acceptor_accuracy', 'neither_accuracy', 'neither_precision', 'neither_recall', 'neither_f1']:
                #         print_dict[k] = f"{v:.4f}"
                # pbar.set_postfix(print_dict)

                ############################
                # Topk SpliceAI assessment approach
                ############################
                print_top_k = False
                if batch_idx % 100 == 0:
                    print_top_k = True
                if print_top_k:
                    print("\n\033[1mTraining set metrics:\033[0m")
                # print("labels: ", labels.shape)
                # print("yp: ", yp.shape)
                Y_true_1 = labels[is_expr, :, 1].flatten().cpu().detach().numpy()
                Y_true_2 = labels[is_expr, :, 2].flatten().cpu().detach().numpy()
                Y_pred_1 = yp[is_expr, :, 1].flatten().cpu().detach().numpy()
                Y_pred_2 = yp[is_expr, :, 2].flatten().cpu().detach().numpy()
                acceptor_top1, acceptor_auprc = print_topl_statistics(np.asarray(Y_true_1),
                                      np.asarray(Y_pred_1), metric_files["acceptor_topk"], type='acceptor', print_top_k=print_top_k)
                donor_top1, donor_auprc = print_topl_statistics(np.asarray(Y_true_2),
                                    np.asarray(Y_pred_2), metric_files["donor_topk"], type='donor', print_top_k=print_top_k)

                # Update progress bar
                print_dict = {}
                print_dict["loss"] = metrics["loss"]
                print_dict["acceptor_top1"] = f"{acceptor_top1:.4f}"
                print_dict["acceptor_auprc"] = f"{acceptor_auprc:.4f}"
                print_dict["acceptor_precision"] = f"{metrics['acceptor_precision']:.4f}"
                print_dict["donor_top1"] = f"{donor_top1:.4f}"
                print_dict["donor_auprc"] = f"{donor_auprc:.4f}"
                print_dict["donor_precision"] = f"{metrics['donor_precision']:.4f}"
                pbar.set_postfix(print_dict)

                if print_top_k:
                    print("\n\n")
            pbar.update(1)
            batch_idx += 1
        pbar.close()


def main():
    # os.environ['WANDB_MODE'] = 'disabled'
    # wandb.init(project='spliceai-general', reinit=True)
    chunk_size = sys.argv[1]
    flanking_size = sys.argv[2]
    exp_num = sys.argv[3]
    assert int(flanking_size) in [80, 400, 2000, 10000]
    BATCH_SIZE = 18*2
    device = setup_device()
    # target = "RefSeq_train_test_dataset_SpliceAI"
    target = "train_test_dataset_SpliceAI"
    data_dir, model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths(chunk_size, flanking_size, exp_num)
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
    test_idxs = np.arange(len(test_h5f.keys()) // 2)

    # print("train_idxs: ", train_idxs)
    # print("val_idxs: ", val_idxs)
    # print("test_idxs: ", test_idxs)


    # # train_loader, valid_loader, test_loader = load_data(data_dir, model_output_base, chunk_size, flanking_size)
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

    EPOCH_NUM = 10
    for epoch in range(EPOCH_NUM):
        print("\n--------------------------------------------------------------")
        print(f">> Epoch {epoch + 1}")
        start_time = time.time()

        train(model, train_h5f, train_idxs, BATCH_SIZE, criterion, optimizer, scheduler, device, params, train_metric_files)
        validate_test_model(model, train_h5f, val_idxs, BATCH_SIZE, criterion, device, params, valid_metric_files, "validation")
        validate_test_model(model, test_h5f, test_idxs, BATCH_SIZE, criterion, device, params, test_metric_files, "test")
        torch.save(model.state_dict(), f"{model_output_base}/model_{epoch}.pt")
        # model.save(f"{model_output_base}/model_{epoch}.pt")
    
        print("--- %s seconds ---" % (time.time() - start_time))
        print("--------------------------------------------------------------")
    train_h5f.close()
    test_h5f.close()

if __name__ == "__main__":
    main()
