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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import wandb


def setup_device():
    """Select computation device based on availability."""
    device_str = "cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" else "cpu"
    return torch.device(device_str)


def initialize_paths(output_dir, project_name, flanking_size, exp_num, sequence_length, model_arch, loss_fun, random_seed):
    """Initialize project directories and create them if they don't exist."""
    ####################################
    # Modify the model verson here!!
    ####################################
    MODEL_VERSION = f"{model_arch}_{loss_fun}_{project_name}_{flanking_size}_{exp_num}_rs{random_seed}"
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
    model = SpliceAI(L, W, AR).to(device)
    print(model, file=sys.stderr)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 7, 8, 9], gamma=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    # Replace the existing scheduler with ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, 1000, len(train_loader)*EPOCH_NUM)
    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE}
    return model, None, optimizer, scheduler, params


def classwise_accuracy(true_classes, predicted_classes, num_classes):
    class_accuracies = []
    for i in range(num_classes):
        true_positives = np.sum((predicted_classes == i) & (true_classes == i))
        total_class_samples = np.sum(true_classes == i)
        if total_class_samples > 0:
            accuracy = true_positives / total_class_samples
        else:
            accuracy = 0.0  # Or set to an appropriate value for classes with no samples
        class_accuracies.append(accuracy)
    return class_accuracies


def metrics(batch_ypred, batch_ylabel, metric_files, run_mode):
    # Convert softmax probabilities to predicted classes
    _, predicted_classes = torch.max(batch_ypred, 1)  # Ensure this matches your data shape correctly
    true_classes = torch.argmax(batch_ylabel, dim=1)  # Adjust the axis if necessary
    # Convert tensors to numpy for compatibility with scikit-learn
    true_classes = true_classes.numpy()
    predicted_classes = predicted_classes.numpy()
    # Flatten arrays if they're 2D (for multi-class, not multi-label)
    true_classes_flat = true_classes.flatten()
    predicted_classes_flat = predicted_classes.flatten()
    # Now, calculate the metrics without iterating over each class
    accuracy = accuracy_score(true_classes_flat, predicted_classes_flat)
    precision, recall, f1, _ = precision_recall_fscore_support(true_classes_flat, predicted_classes_flat, average=None)
    class_accuracies = classwise_accuracy(true_classes, predicted_classes, 3)
    overall_accuracy = np.mean(class_accuracies)
    print(f"Overall Accuracy: {overall_accuracy}")
    for k, v in metric_files.items():
        with open(v, 'a') as f:
            if k == "accuracy":
                f.write(f"{overall_accuracy}\n")
    ss_types = ["Non-splice", "acceptor", "donor"]
    for i, (acc, prec, rec, f1_score) in enumerate(zip(class_accuracies, precision, recall, f1)):
        print(f"Class {ss_types[i]}\t: Accuracy={acc}, Precision={prec}, Recall={rec}, F1={f1_score}")
        if ss_types[i] == "Non-splice":
            continue
        for k, v in metric_files.items():
            with open(v, 'a') as f:
                if k == f"{ss_types[i]}_precision":
                    f.write(f"{prec}\n")
                elif k == f"{ss_types[i]}_recall":
                    f.write(f"{rec}\n")
                elif k == f"{ss_types[i]}_f1":
                    f.write(f"{f1_score}\n")
                elif k == f"{ss_types[i]}_accuracy":
                    f.write(f"{acc}\n")
        wandb.log({
            f'{run_mode}/{ss_types[i]} precision': prec,
            f'{run_mode}/{ss_types[i]} recall': rec,
            f'{run_mode}/{ss_types[i]} F1': f1_score,
            f'{run_mode}/{ss_types[i]} accuracy': acc
        })


def load_data_from_shard(h5f, shard_idx, device, batch_size, params, shuffle=False):
    X = h5f[f'X{shard_idx}'][:].transpose(0, 2, 1)
    Y = h5f[f'Y{shard_idx}'][0, ...].transpose(0, 2, 1)
    # print("\n\tX.shape: ", X.shape)
    # print("\tY.shape: ", Y.shape)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    ds = TensorDataset(X, Y)
    # print("\rds: ", ds)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, pin_memory=True)
    # return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=8, pin_memory=True)


def model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode, criterion):
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
        batch_ylabel = batch_ylabel[is_expr][subset_indices, :, :]
        batch_ypred = batch_ypred[is_expr][subset_indices, :, :]
        Y_true_1 = batch_ylabel[:, 1, :].flatten().cpu().detach().numpy()
        Y_true_2 = batch_ylabel[:, 2, :].flatten().cpu().detach().numpy()
        Y_pred_1 = batch_ypred[:, 1, :].flatten().cpu().detach().numpy()
        Y_pred_2 = batch_ypred[:, 2, :].flatten().cpu().detach().numpy()
        acceptor_topk_accuracy, acceptor_auprc = print_topl_statistics(np.asarray(Y_true_1),
                            np.asarray(Y_pred_1), metric_files["acceptor_topk"], type='acceptor', print_top_k=True)
        donor_topk_accuracy, donor_auprc = print_topl_statistics(np.asarray(Y_true_2),
                            np.asarray(Y_pred_2), metric_files["donor_topk"], type='donor', print_top_k=True)
        if criterion == "cross_entropy_loss":
            loss = categorical_crossentropy_2d(batch_ylabel, batch_ypred)
        elif criterion == "focal_loss":
            loss = focal_loss(batch_ylabel, batch_ypred)
        for k, v in metric_files.items():
            with open(v, 'a') as f:
                if k == "loss_batch":
                    f.write(f"{loss.item()}\n")
                elif k == "donor_topk":
                    f.write(f"{donor_topk_accuracy}\n")
                elif k == "donor_auprc":
                    f.write(f"{donor_auprc}\n")
                elif k == "acceptor_topk":
                    f.write(f"{acceptor_topk_accuracy}\n")
                elif k == "acceptor_auprc":
                    f.write(f"{acceptor_auprc}\n")
                elif k == "acceptor_auroc":
                    f.write(f"{acceptor_auroc}\n")
        wandb.log({
            f'{run_mode}/Loss batch': loss.item(),
            f'{run_mode}/acceptor TopK': acceptor_topk_accuracy,
            f'{run_mode}/donor TopK': donor_topk_accuracy,
            f'{run_mode}/acceptor AUPRC': acceptor_auprc,
            f'{run_mode}/donor AUPRC': donor_auprc,
        })
        print("***************************************\n\n")
        metrics(batch_ypred, batch_ylabel, metric_files, run_mode)
    batch_ylabel = []
    batch_ypred = []
    return loss


def valid_epoch(model, h5f, idxs, batch_size, criterion, device, params, metric_files, run_mode, sample_freq):
    print(f"\033[1m{run_mode.capitalize()}ing model...\033[0m")
    model.eval()
    running_loss = 0.0
    np.random.seed(params["RANDOM_SEED"])  # You can choose any number as a seed
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
            # wandb.log({
            #     f'{run_mode}/loss_every_update': loss.item(),
            # })
            running_loss += loss.item()
            # print("loss: ", loss.item())
            batch_ylabel.append(labels.detach().cpu())
            batch_ypred.append(yp.detach().cpu())
            print_dict["loss"] = loss.item()
            pbar.set_postfix(print_dict)
            pbar.update(1)
            batch_idx += 1
        pbar.close()
    eval_loss = model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode, criterion)
    return eval_loss


def train_epoch(model, h5f, idxs, batch_size, criterion, optimizer, scheduler, device, params, metric_files, run_mode, sample_freq):
    print(f"\033[1m{run_mode.capitalize()}ing model...\033[0m")
    model.train()
    running_loss = 0.0
    np.random.seed(params["RANDOM_SEED"])  # You can choose any number as a seed
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
            # wandb.log({
            #     f'{run_mode}/loss_every_update': loss.item(),
            # })
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
    eval_loss = model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode, criterion)
    return eval_loss


def train(args):
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
    model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths(output_dir, project_name, flanking_size, exp_num, sequence_length, model_arch, args.loss, args.random_seed)
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
    RANDOM_SEED = args.random_seed
    print("Batch_num: ", batch_num, file=sys.stderr)
    print("RANDOM_SEED: ", RANDOM_SEED, file=sys.stderr)
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
    params["RANDOM_SEED"] = RANDOM_SEED
    train_metric_files = {
        'donor_topk_all': f'{log_output_train_base}/donor_topk_all.txt',
        'donor_topk': f'{log_output_train_base}/donor_topk.txt',
        'donor_auprc': f'{log_output_train_base}/donor_auprc.txt',
        'donor_accuracy': f'{log_output_train_base}/donor_accuracy.txt',
        'donor_precision': f'{log_output_train_base}/donor_precision.txt',
        'donor_recall': f'{log_output_train_base}/donor_recall.txt',
        'donor_f1': f'{log_output_train_base}/donor_f1.txt',
        'acceptor_topk_all': f'{log_output_train_base}/acceptor_topk_all.txt',
        'acceptor_topk': f'{log_output_train_base}/acceptor_topk.txt',
        'acceptor_auprc': f'{log_output_train_base}/acceptor_auprc.txt',
        'acceptor_accuracy': f'{log_output_train_base}/acceptor_accuracy.txt',
        'acceptor_precision': f'{log_output_train_base}/acceptor_precision.txt',
        'acceptor_recall': f'{log_output_train_base}/acceptor_recall.txt',
        'acceptor_f1': f'{log_output_train_base}/acceptor_f1.txt',
        'prc': f'{log_output_train_base}/prc.png',
        'accuracy': f'{log_output_train_base}/accuracy.txt',
        'loss_batch': f'{log_output_train_base}/loss_batch.txt',
        'loss_every_update': f'{log_output_train_base}/loss_every_update.txt',
    }
    valid_metric_files = {
        'donor_topk_all': f'{log_output_val_base}/donor_topk_all.txt',
        'donor_topk': f'{log_output_val_base}/donor_topk.txt',
        'donor_auprc': f'{log_output_val_base}/donor_auprc.txt',
        'donor_accuracy': f'{log_output_val_base}/donor_accuracy.txt',
        'donor_precision': f'{log_output_val_base}/donor_precision.txt',
        'donor_recall': f'{log_output_val_base}/donor_recall.txt',
        'donor_f1': f'{log_output_val_base}/donor_f1.txt',
        'acceptor_topk_all': f'{log_output_val_base}/acceptor_topk_all.txt',
        'acceptor_topk': f'{log_output_val_base}/acceptor_topk.txt',
        'acceptor_auprc': f'{log_output_val_base}/acceptor_auprc.txt',
        'acceptor_accuracy': f'{log_output_val_base}/acceptor_accuracy.txt',
        'acceptor_precision': f'{log_output_val_base}/acceptor_precision.txt',
        'acceptor_recall': f'{log_output_val_base}/acceptor_recall.txt',
        'acceptor_f1': f'{log_output_val_base}/acceptor_f1.txt',
        'prc': f'{log_output_val_base}/prc.png',
        'accuracy': f'{log_output_val_base}/accuracy.txt',
        'loss_batch': f'{log_output_val_base}/loss_batch.txt',
        'loss_every_update': f'{log_output_val_base}/loss_every_update.txt',
    }
    test_metric_files = {
        'donor_topk_all': f'{log_output_test_base}/donor_topk_all.txt',
        'donor_topk': f'{log_output_test_base}/donor_topk.txt',
        'donor_auprc': f'{log_output_test_base}/donor_auprc.txt',
        'donor_accuracy': f'{log_output_test_base}/donor_accuracy.txt',
        'donor_precision': f'{log_output_test_base}/donor_precision.txt',
        'donor_recall': f'{log_output_test_base}/donor_recall.txt',
        'donor_f1': f'{log_output_test_base}/donor_f1.txt',
        'acceptor_topk_all': f'{log_output_test_base}/acceptor_topk_all.txt',
        'acceptor_topk': f'{log_output_test_base}/acceptor_topk.txt',
        'acceptor_auprc': f'{log_output_test_base}/acceptor_auprc.txt',
        'acceptor_accuracy': f'{log_output_test_base}/acceptor_accuracy.txt',
        'acceptor_precision': f'{log_output_test_base}/acceptor_precision.txt',
        'acceptor_recall': f'{log_output_test_base}/acceptor_recall.txt',
        'acceptor_f1': f'{log_output_test_base}/acceptor_f1.txt',
        'prc': f'{log_output_test_base}/prc.png',        
        'accuracy': f'{log_output_test_base}/accuracy.txt',
        'loss_batch': f'{log_output_test_base}/loss_batch.txt',
        'loss_every_update': f'{log_output_test_base}/loss_every_update.txt',
    }
    SAMPLE_FREQ = 1000
    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_patience = 10  # For example, stop after 10 epochs with no improvement
    for epoch in range(1000):
        print("\n--------------------------------------------------------------")
        # Print the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f">> Epoch {epoch + 1}; Current Learning Rate: {current_lr}")
        wandb.log({
            f'train/learning_rate': current_lr,
        })
        start_time = time.time()
        train_loss = train_epoch(model, train_h5f, train_idxs, params["BATCH_SIZE"], args.loss, optimizer, scheduler, device, params, train_metric_files, run_mode="train", sample_freq=SAMPLE_FREQ)
        val_loss = valid_epoch(model, train_h5f, val_idxs, params["BATCH_SIZE"], args.loss, device, params, valid_metric_files, run_mode="validation", sample_freq=SAMPLE_FREQ)
        test_loss = valid_epoch(model, test_h5f, test_idxs, params["BATCH_SIZE"], args.loss, device, params, test_metric_files, run_mode="test", sample_freq=SAMPLE_FREQ)
        # Scheduler step with validation loss
        scheduler.step(val_loss)
        # Check for early stopping or model improvement
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            # Consider saving the best model here
            torch.save(model.state_dict(), f"{model_output_base}/best_model.pt")
            print("New best model saved.")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epochs.")
            if epochs_no_improve >= n_patience:
                print("Early stopping triggered.")
                break  # Break out of the loop to stop training
        # torch.save(model.state_dict(), f"{model_output_base}/model_{epoch}.pt")
        print("--- %s seconds ---" % (time.time() - start_time))
        print("--------------------------------------------------------------")
    train_h5f.close()
    test_h5f.close()