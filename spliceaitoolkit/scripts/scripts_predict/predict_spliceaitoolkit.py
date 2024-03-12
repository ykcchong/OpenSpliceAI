import argparse
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import platform
from spliceai import *
from utils import *
from constants import *
import h5py
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from keras.models import load_model
import wandb

RANDOM_SEED = 42

def setup_device():
    """Select computation device based on availability."""
    device_str = "cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" else "cpu"
    return torch.device(device_str)


def initialize_paths(output_dir, project_name, flanking_size, sequence_length):
    """Initialize project directories and create them if they don't exist."""
    ####################################
    # Modify the model verson here!!
    ####################################
    MODEL_VERSION = f"{project_name}_{sequence_length}_{flanking_size}"
    ####################################
    # Modify the model verson here!!
    ####################################
    model_train_outdir = f"{output_dir}/{MODEL_VERSION}/"
    model_output_base = f"{model_train_outdir}models/"
    log_output_base = f"{model_train_outdir}LOG/"
    log_output_test_base = f"{log_output_base}TEST/"
    for path in [model_output_base, log_output_test_base]:
        os.makedirs(path, exist_ok=True)
    return model_output_base, log_output_test_base


def calculate_metrics(y_true, y_pred):
    """Calculate metrics including precision, recall, f1-score, and accuracy."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy


def threshold_predictions(y_probs, threshold=0.5):
    """Threshold probabilities to get binary predictions."""
    return (y_probs > threshold).astype(int)


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


def metrics(batch_ypred, batch_ylabel, metric_files):
    # # Placeholder for AUC and AUPRC scores
    # auc_scores = []
    # auprc_scores = []
    # # Placeholder for ROC and PRC curves
    # roc_curves = []
    # prc_curves = []
    # # Assuming batch_ypred and batch_ylabel are your tensors
    # # Convert probabilities to predicted classes
    # pred_classes = torch.argmax(batch_ypred, dim=1)
    # # If batch_ylabel is not already in class format, convert it
    # true_classes = torch.argmax(batch_ylabel, dim=1)
    # # Convert tensors to numpy arrays for compatibility with scikit-learn
    # pred_classes_np = pred_classes.numpy()
    # true_classes_np = true_classes.numpy()

    # num_classes = 3



    # Assuming batch_ypred and batch_ylabel are your tensors
    # Convert tensors to numpy arrays for compatibility with scikit-learn
    y_pred_numpy = batch_ypred.numpy()
    y_true_numpy = batch_ylabel.numpy()

    num_classes = 3

    # Ensure labels are in the correct shape for label_binarize
    # Flatten y_true_numpy if it's not already 1D (assuming it's one-hot encoded)
    if y_true_numpy.ndim > 1:
        y_true_numpy = np.argmax(y_true_numpy, axis=1)

    y_true_binarized = label_binarize(y_true_numpy, classes=range(num_classes))

    # Calculate metrics
    roc_auc = roc_auc_score(y_true_binarized, y_pred_numpy, multi_class='ovr', average='macro')
    print("Average AUROC (One-vs-Rest):", roc_auc)

    average_precision = average_precision_score(y_true_binarized, y_pred_numpy, average='macro')
    print("Average AUPRC:", average_precision)

    # Compute and plot ROC curves for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.figure()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_numpy[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc="lower right")
    plt.savefig(metric_files['roc'])  # Assuming 'roc' key in metric_files dictionary holds a correct file path



    # precision = np.zeros(num_classes)
    # recall = np.zeros(num_classes)
    # f1 = np.zeros(num_classes)
    # accuracies = torch.zeros(num_classes)

    # # Binarize the output labels for multi-class AUROC
    # y_true_binarized = label_binarize(true_classes_np, classes=range(num_classes))
    # # Calculate the ROC AUC score
    # # This computes the average AUC for each class against all others
    # roc_auc = roc_auc_score(y_true_binarized, batch_ypred.numpy(), multi_class='ovr')
    # print("Average AUROC (One-vs-Rest):", roc_auc)


    # # Calculate the AUPRC for each class and average them
    # average_precision = average_precision_score(y_true_binarized, batch_ypred.numpy(), average='macro')
    # print("Average AUPRC:", average_precision)

    # for i in range(num_classes):
    #     class_indices = true_classes == i
    #     correct_predictions = pred_classes[class_indices] == true_classes[class_indices]
    #     accuracies[i] = correct_predictions.float().mean()
        
    #     # Binary labels for the current class
    #     true_binary = (true_classes_np == i).astype(int)
    #     pred_probs_i = batch_ypred.numpy()[:, i]

    #     # Calculate ROC curve and AUC
    #     fpr, tpr, _ = roc_curve(true_binary, pred_probs_i)
    #     auc_scores.append(roc_auc)
    #     roc_curves.append((fpr, tpr, roc_auc))

    #     # Calculate Precision-Recall curve and AUPRC
    #     precision, recall, _ = precision_recall_curve(true_binary, pred_probs_i)
    #     pr_auc = average_precision_score(true_binary, pred_probs_i)
    #     auprc_scores.append(pr_auc)
    #     prc_curves.append((precision, recall, pr_auc))
    # print("accuracies: ", accuracies)
    # print("precision: ", precision)
    # print("recall: ", recall)
    # print("f1: ", f1)

    # # Compute ROC curve and ROC area for each class
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()

    # for i in range(num_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], batch_ypred.numpy()[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])

    # # Plotting
    # plt.figure()
    # colors = ['aqua', 'darkorange', 'cornflowerblue']
    # for i, color in zip(range(num_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=2,
    #             label='ROC curve of class {0} (area = {1:0.2f})'
    #             ''.format(i, roc_auc[i]))

    # plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    # plt.legend(loc="lower right")
    # plt.savefig(metric_files['roc.png'])

    # # Plotting ROC Curves
    # plt.figure(figsize=(10, 5))
    # for i, (fpr, tpr, roc_auc) in enumerate(roc_curves):
    #     plt.plot(fpr, tpr, lw=2, label='Class %d (area = %0.2f)' % (i, roc_auc))
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) curves')
    # plt.legend(loc="lower right")
    # plt.savefig(metric_files['roc.png'])
    # # Plotting Precision-Recall Curves
    # plt.figure(figsize=(10, 5))
    # for i, (precision, recall, pr_auc) in enumerate(prc_curves):
    #     plt.plot(recall, precision, lw=2, label='Class %d (area = %0.2f)' % (i, pr_auc))
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall curves')
    # plt.legend(loc="lower left")
    # plt.savefig(metric_files['prc.png'])


def model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode):
    batch_ylabel = torch.cat(batch_ylabel, dim=0)
    batch_ypred = torch.cat(batch_ypred, dim=0)
    is_expr = (batch_ylabel.sum(axis=(1,2)) >= 1).cpu().numpy()
    print("batch_ylabel: ", batch_ylabel.shape)
    print("batch_ypred: ", batch_ypred.shape)
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

        metrics(batch_ypred, batch_ylabel, metric_files)
        # if criterion == "cross_entropy_loss":
        loss = categorical_crossentropy_2d(batch_ylabel, batch_ypred)
        # elif criterion == "focal_loss":
        #     loss = focal_loss(batch_ylabel, batch_ypred)
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


def valid_epoch(model, h5f, idxs, batch_size, device, params, metric_files, run_mode, sample_freq):
    print(f"\033[1m{run_mode.capitalize()}ing model...\033[0m")
    model.eval()
    running_loss = 0.0
    np.random.seed(RANDOM_SEED)  # You can choose any number as a seed
    # shuffled_idxs = np.random.choice(idxs, size=len(idxs), replace=False)    
    shuffled_idxs = idxs
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
            # if criterion == "cross_entropy_loss":
            loss = categorical_crossentropy_2d(labels, yp)
            # elif criterion == "focal_loss":
            #     loss = focal_loss(labels, yp)
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
            # if batch_idx % sample_freq == 0:
            #     model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode)        
        if i == 5:
            break
        pbar.close()
    model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode)


def initialize_model_and_optim(flanking_size):
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
    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE}
    return params


def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', '-p', type=str)
    parser.add_argument('--disable-wandb', '-d', action='store_true', default=False)
    parser.add_argument('--flanking-size', '-f', type=int, default=80)
    parser.add_argument('--test-dataset', '-test', type=str)
    parser.add_argument('--project-name', '-s', type=str)
    parser.add_argument('--model', '-m', default="SpliceAI", type=str)
    args = parser.parse_args()
    print("args: ", args, file=sys.stderr)
    print("Running SpliceAI-toolkit with 'predict' mode")

    
    output_dir = args.output_dir
    project_name = args.project_name
    sequence_length = 5000
    flanking_size = int(args.flanking_size)
    model_path = args.model
    testing_dataset = args.test_dataset
    assert int(flanking_size) in [80, 400, 2000, 10000]
    if args.disable_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    wandb.init(project=f'{project_name}', reinit=True)
    device = setup_device()
    print("output_dir: ", output_dir, file=sys.stderr)
    print("flanking_size: ", flanking_size, file=sys.stderr)
    print("model_path: ", model_path, file=sys.stderr)
    print("testing_dataset: ", testing_dataset, file=sys.stderr)
    print("device: ", device, file=sys.stderr)
    model_output_base, log_output_test_base = initialize_paths(output_dir, project_name, flanking_size, sequence_length)
    print("* Project name: ", args.project_name, file=sys.stderr)
    print("* Model_output_base: ", model_output_base, file=sys.stderr)
    print("* Log_output_test_base: ", log_output_test_base, file=sys.stderr)

    test_h5f = h5py.File(testing_dataset, 'r')
    np.random.seed(RANDOM_SEED)  # You can choose any number as a seed
    test_idxs = np.arange(len(test_h5f.keys()) // 2)
    print("test_idxs: ", test_idxs, file=sys.stderr)


    params = initialize_model_and_optim(flanking_size)


    model = SpliceAI(params['L'], params['W'], params['AR'])
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    # model.eval()
    print("Model: ", model)

    test_metric_files = {
        'topk_donor': f'{log_output_test_base}/donor_topk.txt',
        'auprc_donor': f'{log_output_test_base}/donor_accuracy.txt',
        'topk_acceptor': f'{log_output_test_base}/acceptor_topk.txt',
        'auprc_acceptor': f'{log_output_test_base}/acceptor_accuracy.txt',
        'loss_batch': f'{log_output_test_base}/loss_batch.txt',
        'loss_every_update': f'{log_output_test_base}/loss_every_update.txt',
        'auc': f'{log_output_test_base}/auc.png',
        'prc': f'{log_output_test_base}/prc.png',
    }
    SAMPLE_FREQ = 1000
    print("\n--------------------------------------------------------------")
    start_time = time.time()
    BATCH_SIZE = 36
    valid_epoch(model, test_h5f, test_idxs, BATCH_SIZE, device, params, test_metric_files, run_mode="test", sample_freq=SAMPLE_FREQ)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("--------------------------------------------------------------")
    test_h5f.close()

if __name__ == "__main__":
    predict()