import h5py
import platform
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from math import ceil
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, accuracy_score
from openspliceai.constants import *
import wandb # weights and biases: need to connect to this one

def setup_environment(args):
    assert int(args.flanking_size) in [80, 400, 2000, 10000]
    if not args.enable_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    wandb.init(project=f'{args.project_name}', reinit=True)
    device = setup_device()
    print("device: ", device, file=sys.stderr)
    return device


def initialize_paths(args):
    model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths_inner(
        args.output_dir, args.project_name, args.flanking_size, args.exp_num, SL, args.loss, args.random_seed
    )
    return model_output_base, log_output_train_base, log_output_val_base, log_output_test_base


def load_datasets(args):
    train_h5f = h5py.File(args.train_dataset, 'r')
    test_h5f = h5py.File(args.test_dataset, 'r')
    batch_num = len(train_h5f.keys()) // 2
    print("* Batch_num: ", batch_num, file=sys.stderr)
    return train_h5f, test_h5f, batch_num


def generate_indices(batch_num, random_seed, test_h5f):
    np.random.seed(random_seed)
    # idxs = np.random.permutation(batch_num)
    # train_idxs = idxs[:int(0.9 * batch_num)]
    # val_idxs = idxs[int(0.9 * batch_num):]
    # test_idxs = np.arange(len(test_h5f.keys()) // 2)
    # Generate and shuffle indices for training set
    train_idxs = np.arange(batch_num)
    np.random.shuffle(train_idxs)    
    # Generate indices for test set
    test_idxs = np.arange(len(test_h5f.keys()) // 2)
    np.random.shuffle(test_idxs)
    # Split test set into test and validation
    val_size = int(0.5 * len(test_idxs))  # 10% for validation
    val_idxs = test_idxs[:val_size]
    test_idxs = test_idxs[val_size:]
    return train_idxs, val_idxs, test_idxs


def create_metric_files(log_output_base):
    metric_types = ['donor_topk_all', 'donor_topk', 'donor_auprc', 'donor_accuracy', 'donor_precision', 
                    'donor_recall', 'donor_f1', 'acceptor_topk_all', 'acceptor_topk', 'acceptor_auprc', 
                    'acceptor_accuracy', 'acceptor_precision', 'acceptor_recall', 'acceptor_f1', 
                    'accuracy', 'loss_batch', 'loss_every_update']
    return {metric: f'{log_output_base}/{metric}.txt' for metric in metric_types}


def setup_device():
    device_str = "cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" else "cpu"
    return torch.device(device_str)


def initialize_paths_inner(output_dir, project_name, flanking_size, exp_num, sequence_length, loss_fun, random_seed):
    MODEL_VERSION = f"SpliceAI_{project_name}_{flanking_size}_{exp_num}_rs{random_seed}"
    model_train_outdir = f"{output_dir}/{MODEL_VERSION}/{exp_num}/"
    model_output_base = f"{model_train_outdir}models/"
    log_output_base = f"{model_train_outdir}LOG/"
    log_output_train_base = f"{log_output_base}TRAIN/"
    log_output_val_base = f"{log_output_base}VAL/"
    log_output_test_base = f"{log_output_base}TEST/"
    for path in [model_output_base, log_output_train_base, log_output_val_base, log_output_test_base]:
        os.makedirs(path, exist_ok=True)
    return model_output_base, log_output_train_base, log_output_val_base, log_output_test_base


def load_data_from_shard(h5f, shard_idx, device, batch_size, params, shuffle=False):
    X = h5f[f'X{shard_idx}'][:].transpose(0, 2, 1)
    Y = h5f[f'Y{shard_idx}'][0, ...].transpose(0, 2, 1)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    ds = TensorDataset(X, Y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, pin_memory=True)


def classwise_accuracy(true_classes, predicted_classes, num_classes):
    class_accuracies = []
    for i in range(num_classes):
        true_positives = np.sum((predicted_classes == i) & (true_classes == i))
        total_class_samples = np.sum(true_classes == i)
        if total_class_samples > 0:
            accuracy = true_positives / total_class_samples
        else:
            accuracy = 0.0
        class_accuracies.append(accuracy)
    return class_accuracies


def metrics(batch_ypred, batch_ylabel, metric_files, run_mode):
    _, predicted_classes = torch.max(batch_ypred, 1)
    true_classes = torch.argmax(batch_ylabel, dim=1)
    true_classes = true_classes.numpy()
    predicted_classes = predicted_classes.numpy()
    true_classes_flat = true_classes.flatten()
    predicted_classes_flat = predicted_classes.flatten()
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


def model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode, criterion):
    batch_ylabel = torch.cat(batch_ylabel, dim=0)
    batch_ypred = torch.cat(batch_ypred, dim=0)
    is_expr = (batch_ylabel.sum(axis=(1,2)) >= 1).cpu().numpy()
    if np.any(is_expr):
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
                            np.asarray(Y_pred_1), metric_files["acceptor_topk_all"], type='acceptor', print_top_k=True)
        donor_topk_accuracy, donor_auprc = print_topl_statistics(np.asarray(Y_true_2),
                            np.asarray(Y_pred_2), metric_files["donor_topk_all"], type='donor', print_top_k=True)
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
        wandb.log({
            f'{run_mode}/Loss batch': loss.item(),
            f'{run_mode}/acceptor TopK': acceptor_topk_accuracy,
            f'{run_mode}/donor TopK': donor_topk_accuracy,
            f'{run_mode}/acceptor AUPRC': acceptor_auprc,
            f'{run_mode}/donor AUPRC': donor_auprc,
        })
        print("***************************************\n")
        metrics(batch_ypred, batch_ylabel, metric_files, run_mode)
    batch_ylabel = []
    batch_ypred = []
    return loss


def valid_epoch(model, h5f, idxs, batch_size, criterion, device, params, metric_files, flanking_size, run_mode):
    print(f"\033[1m{run_mode.capitalize()}ing model...\033[0m")
    model.eval()
    running_loss = 0.0
    np.random.seed(params["RANDOM_SEED"])
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
            DNAs, labels = clip_datapoints(DNAs, labels, params["CL"], CL_max, params["N_GPUS"])
            DNAs, labels = DNAs.to(torch.float32).to(device), labels.to(torch.float32).to(device)
            yp = model(DNAs)
            if criterion == "cross_entropy_loss":
                loss = categorical_crossentropy_2d(labels, yp)
            elif criterion == "focal_loss":
                loss = focal_loss(labels, yp)
            with open(metric_files["loss_every_update"], 'a') as f:
                f.write(f"{loss.item()}\n")
            running_loss += loss.item()
            batch_ylabel.append(labels.detach().cpu())
            batch_ypred.append(yp.detach().cpu())
            print_dict["loss"] = loss.item()
            pbar.set_postfix(print_dict)
            pbar.update(1)
            batch_idx += 1
        pbar.close()
    eval_loss = model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode, criterion)
    return eval_loss


def train_epoch(model, h5f, idxs, batch_size, criterion, optimizer, scheduler, device, params, metric_files, flanking_size, run_mode):
    print(f"\033[1m{run_mode.capitalize()}ing model...\033[0m")
    model.train()
    running_loss = 0.0
    np.random.seed(params["RANDOM_SEED"])
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
            DNAs, labels = clip_datapoints(DNAs, labels, params["CL"], CL_max, params["N_GPUS"])
            DNAs, labels = DNAs.to(torch.float32).to(device), labels.to(torch.float32).to(device)
            optimizer.zero_grad()
            yp = model(DNAs)
            if criterion == "cross_entropy_loss":
                loss = categorical_crossentropy_2d(labels, yp)
            elif criterion == "focal_loss":
                loss = focal_loss(labels, yp)
            with open(metric_files["loss_every_update"], 'a') as f:
                f.write(f"{loss.item()}\n")
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_ylabel.append(labels.detach().cpu())
            batch_ypred.append(yp.detach().cpu())
            print_dict["loss"] = loss.item()
            pbar.set_postfix(print_dict)
            pbar.update(1)
            batch_idx += 1
        pbar.close()
    eval_loss = model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode, criterion)
    return eval_loss


def calculate_metrics(y_true, y_pred):
    """Calculate metrics including precision, recall, f1-score, and accuracy."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy


def threshold_predictions(y_probs, threshold=0.5):
    """Threshold probabilities to get binary predictions."""
    return (y_probs > threshold).astype(int)


def clip_datapoints(X, Y, CL, CL_max, N_GPUS):
    """
    This function is necessary to make sure of the following:
    (i) Each time model_m.fit is called, the number of datapoints is a
    multiple of N_GPUS. Failure to ensure this often results in crashes.
    (ii) If the required context length is less than CL_max, then
    appropriate clipping is done below.
    Additionally, Y is also converted to a list (the .h5 files store 
    them as an array).
    """
    rem = X.shape[0]%N_GPUS
    clip = (CL_max-CL)//2
    if rem != 0 and clip != 0:
        return X[:-rem, :, clip:-clip], Y[:-rem]
    elif rem == 0 and clip != 0:
        return X[:, :, clip:-clip], Y
    elif rem != 0 and clip == 0:
        return X[:-rem], Y[:-rem]
    else:
        return X, Y


def print_topl_statistics(y_true, y_pred, file, type='acceptor', print_top_k=False):
    # Prints the following information: top-kL statistics for k=0.5,1,2,4,
    # auprc, thresholds for k=0.5,1,2,4, number of true splice sites.
    idx_true = np.nonzero(y_true == 1)[0]
    argsorted_y_pred = np.argsort(y_pred)
    sorted_y_pred = np.sort(y_pred)
    topkl_accuracy = []
    threshold = []
    for top_length in [0.5, 1, 2, 4]:
        num_elements = int(top_length * len(idx_true))
        if num_elements > len(y_pred):  # Check to prevent out-of-bounds access
            print(f"Warning: Requested top_length {top_length} with {len(idx_true)} true elements exceeds y_pred size of {len(y_pred)}. Adjusting to fit.")
            num_elements = len(y_pred)  # Adjust num_elements to prevent out-of-bounds error
        idx_pred = argsorted_y_pred[-int(top_length*len(idx_true)):]
        topkl_accuracy += [np.size(np.intersect1d(idx_true, idx_pred)) \
                  / float(min(len(idx_pred), len(idx_true))+1e-10)]
        threshold += [sorted_y_pred[-num_elements]]
    auprc = average_precision_score(y_true, y_pred)
    if print_top_k:
        print(f"\n\033[1m{type}:\033[0m")
        print((("%.4f\t\033[91m%.4f\t\033[0m%.4f\t%.4f\t\033[94m%.4f\t\033[0m"
            + "%.4f\t%.4f\t%.4f\t%.4f\t%d") % (topkl_accuracy[0], topkl_accuracy[1], topkl_accuracy[2],
            topkl_accuracy[3], auprc, threshold[0], threshold[1],
            threshold[2], threshold[3], len(idx_true))))
    with open(file, 'a') as f:
        f.write((("%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t"
          + "%.4f\t%.4f\t%.4f\t%.4f\t%d\n") % (topkl_accuracy[0], topkl_accuracy[1], topkl_accuracy[2],
          topkl_accuracy[3], auprc, threshold[0], threshold[1],
          threshold[2], threshold[3], len(idx_true))))
    return topkl_accuracy[1], auprc


def weighted_binary_cross_entropy(output, target, weights=None):    
    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output+1e-10)) + \
               weights[0] * ((1 - target) * torch.log(1 - output+1e-10))
    else:
        loss = target * torch.log(output+1e-10) + (1 - target) * torch.log(1 - output+1e-10)
    return torch.neg(torch.mean(loss))


def categorical_crossentropy_2d(y_true, y_pred):
    """
    Compute 2D categorical cross-entropy loss.

    Parameters:
    - y_true: tensor of true labels.
    - y_pred: tensor of predicted labels.

    Returns:
    - loss: computed categorical cross-entropy loss.
    """
    return - torch.mean(y_true[:, 0, :]*torch.log(y_pred[:, 0, :]+1e-10)
                        + y_true[:, 1, :]*torch.log(y_pred[:, 1, :]+1e-10)
                        + y_true[:, 2, :]*torch.log(y_pred[:, 2, :]+1e-10))


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Compute 2D focal loss.
    
    Parameters:
    - y_true: tensor of true labels.
    - y_pred: tensor of predicted labels.
    - gamma: focusing parameter.
    - alpha: balancing factor.

    Returns:
    - loss: computed focal loss.
    """
    # Ensuring numerical stability
    gamma = 2
    epsilon = 1e-10
    return - torch.mean(y_true[:, 0, :]*torch.log(y_pred[:, 0, :]+epsilon) * torch.pow(torch.sub(1, y_pred[:, 0, :]), gamma)
                        + y_true[:, 1, :]*torch.log(y_pred[:, 1, :]+epsilon) * torch.pow(torch.sub(1, y_pred[:, 1, :]), gamma)
                        + y_true[:, 2, :]*torch.log(y_pred[:, 2, :]+epsilon) * torch.pow(torch.sub(1, y_pred[:, 2, :]), gamma))

def train_model(model, optimizer, scheduler, train_h5f, test_h5f, train_idxs, val_idxs, test_idxs,
                model_output_base, args, device, params, 
                train_metric_files, valid_metric_files, test_metric_files):
    print("train_idxs: ", train_idxs)
    print("val_idxs: ", val_idxs)
    print("test_idxs: ", test_idxs)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f">> Epoch {epoch + 1}; Current Learning Rate: {current_lr}")
        wandb.log({'train/learning_rate': current_lr})
        start_time = time.time()
        train_loss = train_epoch(model, train_h5f, train_idxs, params["BATCH_SIZE"], args.loss, optimizer, 
                                 scheduler, device, params, train_metric_files, args.flanking_size, run_mode="train")
        val_loss = valid_epoch(model, train_h5f, val_idxs, params["BATCH_SIZE"], args.loss, device, 
                               params, valid_metric_files, args.flanking_size, "validation")
        test_loss = valid_epoch(model, test_h5f, test_idxs, params["BATCH_SIZE"], args.loss, device, 
                                params, test_metric_files, args.flanking_size, "test")
        print(f"Training Loss: {train_loss}")
        print(f"Validation Loss: {val_loss}")
        print(f"Testing Loss: {test_loss}")
        torch.save(model.state_dict(), f"{model_output_base}/model_{epoch}.pt")
        if args.early_stopping:
            scheduler.step(val_loss)
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                torch.save(model.state_dict(), f"{model_output_base}/model_best.pt")
                print("New best model saved.")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"No improvement in validation loss for {epochs_no_improve} epochs.")
                if epochs_no_improve >= args.patience:
                    print("Early stopping triggered.")
                    break
        else:
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                torch.save(model.state_dict(), f"{model_output_base}/model_best.pt")
                print("New best model saved.")
        print(f"--- {time.time() - start_time:.2f} seconds ---")
        print("="*60)