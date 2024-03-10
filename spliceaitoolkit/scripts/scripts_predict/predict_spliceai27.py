import argparse
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import platform
# from spliceai import *
from utils import *
from constants import *
import h5py
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
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


# def model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode, criterion):
#     batch_ylabel = torch.cat(batch_ylabel, dim=0)
#     batch_ypred = torch.cat(batch_ypred, dim=0)
#     is_expr = (batch_ylabel.sum(axis=(1,2)) >= 1).cpu().numpy()
#     if np.any(is_expr):
#         ############################
#         # Topk SpliceAI assessment approach
#         ############################
#         subset_size = 1000
#         indices = np.arange(batch_ylabel[is_expr].shape[0])
#         subset_indices = np.random.choice(indices, size=min(subset_size, len(indices)), replace=False)
#         Y_true_1 = batch_ylabel[is_expr][subset_indices, 1, :].flatten().cpu().detach().numpy()
#         Y_true_2 = batch_ylabel[is_expr][subset_indices, 2, :].flatten().cpu().detach().numpy()
#         Y_pred_1 = batch_ypred[is_expr][subset_indices, 1, :].flatten().cpu().detach().numpy()
#         Y_pred_2 = batch_ypred[is_expr][subset_indices, 2, :].flatten().cpu().detach().numpy()
#         acceptor_topkl_accuracy, acceptor_auprc = print_topl_statistics(np.asarray(Y_true_1),
#                             np.asarray(Y_pred_1), metric_files["topk_acceptor"], type='acceptor', print_top_k=True)
#         donor_topkl_accuracy, donor_auprc = print_topl_statistics(np.asarray(Y_true_2),
#                             np.asarray(Y_pred_2), metric_files["topk_donor"], type='donor', print_top_k=True)
#         if criterion == "cross_entropy_loss":
#             loss = categorical_crossentropy_2d(batch_ylabel, batch_ypred)
#         elif criterion == "focal_loss":
#             loss = focal_loss(batch_ylabel, batch_ypred)
#         for k, v in metric_files.items():
#             with open(v, 'a') as f:
#                 if k == "loss_batch":
#                     f.write(f"{loss.item()}\n")
#                 elif k == "topk_acceptor":
#                     f.write(f"{acceptor_topkl_accuracy}\n")
#                 elif k == "topk_donor":
#                     f.write(f"{donor_topkl_accuracy}\n")
#                 elif k == "auprc_acceptor":
#                     f.write(f"{acceptor_auprc}\n")
#                 elif k == "auprc_donor":
#                     f.write(f"{donor_auprc}\n")
#         wandb.log({
#             f'{run_mode}/loss_batch': loss.item(),
#             f'{run_mode}/topk_acceptor': acceptor_topkl_accuracy,
#             f'{run_mode}/topk_donor': donor_topkl_accuracy,
#             f'{run_mode}/auprc_acceptor': acceptor_auprc,
#             f'{run_mode}/auprc_donor': donor_auprc,
#         })
#         print("***************************************\n\n")
#     batch_ylabel = []
#     batch_ypred = []

def valid_epoch(model, h5f, idxs, batch_size, device, params, metric_files, run_mode, sample_freq):
    print(f"\033[1m{run_mode.capitalize()}ing model...\033[0m")
    print("--------------------------------------------------------------")
    print("\n\033[1mValidation set metrics:\033[0m")

    Y_true_1 = [[] for t in range(1)]
    Y_true_2 = [[] for t in range(1)]
    Y_pred_1 = [[] for t in range(1)]
    Y_pred_2 = [[] for t in range(1)]

    for idx in idxs[:10]:
        X = h5f['X' + str(idx)]
        # [:100]
        Y = h5f['Y' + str(idx)]
        # [:,:100]
        print("\n\tX.shape: ", X.shape)
        print("\tY.shape: ", Y.shape)

        Xc, Yc = clip_datapoints_spliceai27(X, Y, params['CL'], 2)
        # print("\n\tXc.shape: ", Xc.shape)
        # print("\tYc[0].shape: ", Yc[0].shape)
        Yp = model.predict(Xc, batch_size=params['BATCH_SIZE'])

        # loss = categorical_crossentropy_2d(Yc[0], Yp)
        # print("Loss: ", loss)
        if not isinstance(Yp, list):
            Yp = [Yp]

        for t in range(1):
            is_expr = (Yc[t].sum(axis=(1,2)) >= 1)
            Y_true_1[t].extend(Yc[t][is_expr, :, 1].flatten())
            Y_true_2[t].extend(Yc[t][is_expr, :, 2].flatten())
            Y_pred_1[t].extend(Yp[t][is_expr, :, 1].flatten())
            Y_pred_2[t].extend(Yp[t][is_expr, :, 2].flatten())

    print("\n\033[1mAcceptor:\033[0m")
    for t in range(1):
        acceptor_topkl_accuracy, acceptor_auprc = print_topl_statistics(np.asarray(Y_true_1[t]), np.asarray(Y_pred_1[t]), metric_files["topk_acceptor"], type='acceptor', print_top_k=True)

    print("\n\033[1mDonor:\033[0m")
    for t in range(1):
        donor_topkl_accuracy, donor_auprc = print_topl_statistics(np.asarray(Y_true_2[t]),
                                np.asarray(Y_pred_2[t]), metric_files["topk_donor"], type='donor', print_top_k=True)
    print("--------------------------------------------------------------")


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
    # models = [load_model(resource_filename('spliceai', x)) for x in paths]
    model = load_model(model_path)
    print("Model: ", model)

    test_metric_files = {
        'topk_donor': f'{log_output_test_base}/donor_topk.txt',
        'auprc_donor': f'{log_output_test_base}/donor_accuracy.txt',
        'topk_acceptor': f'{log_output_test_base}/acceptor_topk.txt',
        'auprc_acceptor': f'{log_output_test_base}/acceptor_accuracy.txt',
        'loss_batch': f'{log_output_test_base}/loss_batch.txt',
        'loss_every_update': f'{log_output_test_base}/loss_every_update.txt'
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