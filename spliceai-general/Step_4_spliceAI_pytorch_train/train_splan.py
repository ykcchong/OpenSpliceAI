import logging
import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
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

from torch.utils.data import Dataset
from utils import clip_datapoints, print_topl_statistics


def setup_device():
    """Select computation device based on availability."""
    device_str = "cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" else "cpu"
    return torch.device(device_str)


def initialize_paths(chunk_size, flanking_size):
    """Initialize project directories and create them if they don't exist."""
    MODEL_VERSION = f"splan_{chunk_size}chunk_{flanking_size}flank_spliceai_architecture"
    project_root = "/Users/chaokuan-hao/Documents/Projects/spliceAI-MANE/"
    data_dir = f"{project_root}results/gene_sequences_and_labels/"
    model_train_outdir = f"{project_root}results/model_train_outdir/{MODEL_VERSION}/"
    model_output_base = f"{model_train_outdir}models/"
    log_output_base = f"{model_train_outdir}LOG/"
    log_output_train_base = f"{log_output_base}TRAIN/"
    log_output_val_base = f"{log_output_base}VAL/"
    log_output_test_base = f"{log_output_base}TEST/"
    for path in [model_output_base, log_output_train_base, log_output_val_base, log_output_test_base]:
        os.makedirs(path, exist_ok=True)
    return data_dir, model_output_base, log_output_train_base, log_output_val_base, log_output_test_base


def initialize_model_and_optim(device, train_size, flanking_size):
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
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 1000, train_size * EPOCH_NUM)
    return model, criterion, optimizer, scheduler


def filter_dataset_by_shape(dataset, desired_shape=(1300, 4)):
    """
    Filters out items from the dataset that do not match the desired shape.
    
    Parameters:
        dataset (Dataset): The dataset to filter.
        desired_shape (tuple): The shape to retain in the dataset.
    
    Returns:
        Dataset: A new dataset containing only items of the desired shape.
    """
    filtered_indices = [i for i, (seq, _) in enumerate(dataset) if seq.shape == desired_shape]
    return torch.utils.data.Subset(dataset, filtered_indices)




class H5Dataset(Dataset):
    def __init__(self, file_path, idxs):
        """
        Args:
            file_path (string): Path to the h5 file with the dataset.
            idxs (array-like): Indices of the samples to be included in this dataset.
        """
        self.h5_file = file_path
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    # def __getitem__(self, index):
    #     with h5py.File(self.h5_file, 'r') as h5f:
    #         # Assuming 'X' and 'Y' are your dataset keys and they match the indexing logic
    #         X = h5f[f'X{self.idxs[index]}'][:]
    #         Y = h5f[f'Y{self.idxs[index]}'][:]
    #         return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long)



    def __getitem__(self, index):
        with h5py.File(self.h5_file, 'r') as h5f:
            X = h5f[f'X{self.idxs[index]}'][:]
            Y = h5f[f'Y{self.idxs[index]}'][:]
            # print("X.shape: ", X.shape)
            # print("Y.shape: ", len(Y[0]))
            # for i in range(len(Y)):
            #     print(X[i:i+1, :, :].shape, Y[0][i:i+1].shape)
                # return torch.tensor(X[i:i+1, :, :], dtype=torch.float32), torch.tensor(Y[i:i+1], dtype=torch.long)
            
            # # Adjusting the size of X to [1, 15000, 4]
            # if X.shape[0] > 1:
            #     X = X[:1, :, :]  # Select the first [1, 15000, 4] slice
            # elif X.shape[0] < 1:
            #     # Handle the case where there are fewer than 15000 points, if necessary
            #     pass  # Implement padding or other logic as required

            return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long)

def get_loaders(training_dataset, testing_dataset, idx_train, idx_valid, batch_size=32):
    # Instantiate the dataset for both training and validation
    # print("idx_train: ", idx_train)
    # print("idx_valid: ", idx_valid) 
    train_dataset = H5Dataset(training_dataset, idx_train)
    valid_dataset = H5Dataset(testing_dataset, idx_valid)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # print("train_loader: ", len(train_loader))
    # print("valid_loader: ", len(valid_loader))

    return train_loader, valid_loader








def load_data(data_dir, model_output_base, chunk_size, flanking_size):
    """Load and split the dataset into training and testing sets."""
    # stats = data_dir / "stats.txt"
    # dataset = gene_dataset_chunk.GeneDataset(str(data_dir), str(stats))
    # train_size = int(TRAIN_RATIO * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print("data_dir: ", data_dir)
    print("chunk_size: ", chunk_size)
    print("flanking_size: ", flanking_size)
    training_dataset = f"{data_dir}dataset_train.h5"
    testing_dataset = f"{data_dir}dataset_test.h5"

    # train_loader, valid_loader = get_loaders(training_dataset, testing_dataset, idx_train, idx_valid)

    print("training_dataset: ", training_dataset)
    print("testing_dataset: ", testing_dataset)
    h5f = h5py.File(training_dataset, 'r')
    num_idx = len(list(h5f.keys()))//2
    idx_all = np.random.permutation(num_idx)
    idx_train = idx_all[:int(0.9*num_idx)]
    idx_valid = idx_all[int(0.9*num_idx):]#int(0.2*num_idx)]
    print("Number of training datasets: ", len(idx_train))
    print("Number of validation datasets: ", len(idx_valid))
    for idx in idx_train:
        print("idx: ", idx)
    # train_loader, valid_loader = get_loaders(training_dataset, testing_dataset, idx_train, idx_valid)
    # EPOCH_NUM = 10*len(idx_train)
    
    # experiment = f"SpliceAI_{chunk_size}chunk_{flanking_size}flank_MANE_exp"
    # output_dir = f'{model_output_base}/{experiment}/{sys.argv[2]}/'
    # os.makedirs(output_dir, exist_ok=True)
    # files = {name: open(f'{output_dir}{name}_results.txt', 'w') for name in ['training', 'training_loss', 'validation', 'validation_loss']}
    
    # start_time = time.time()
    # for epoch_num in range(EPOCH_NUM):
    #     pass
    #     print("Epoch number: ", epoch_num)
    #     idx = np.random.choice(idx_train)
    #     X = h5f['X' + str(idx)][:]
    #     Y = h5f['Y' + str(idx)][:]
    #     print("X.shape: ", X.shape)
    #     print("Y.shape: ", len(Y[0]))
    #     Xc, Yc = clip_datapoints(X, Y, CL, 2) 
    # for file in files.values():
    #     file.close()
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print("--------------------------------------------------------------")
    # h5f.close()





    # train_dataset = torch.load(os.path.join(data_dir, f'train_dataset_{chunk_size}_{flanking_size}.pth'))
    # test_dataset = torch.load(os.path.join(data_dir, f'testing_dataset_{chunk_size}_{flanking_size}.pth'))

    # train_dataset = filter_dataset_by_shape(train_dataset)
    # test_dataset = filter_dataset_by_shape(test_dataset)
    # print("Train size: ", len(train_dataset))
    # print("Test size: ", len(test_dataset))
    # # for i in train_dataset:
    # #     print("i: ", i)
    # # torch.save(train_dataset, os.path.join(data_dir, 'train_dataset.pth'))
    # # torch.save(test_dataset, os.path.join(data_dir, 'testing_dataset.pth'))

    # # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # # Determine sizes of the datasets
    # num_train = len(train_dataset)
    # indices = list(range(num_train))
    # # Optionally shuffle indices here if your dataset ordering might introduce bias
    # np.random.shuffle(indices)
    # # Using all indices for training in this example, but you could split for validation
    
    # train_size = int(0.8 * num_train)
    # valid_size = int(0.05 * num_train)
    # train_indices = indices[:train_size]  # If you want a validation split, you can divide indices here
    # valid_indices = indices[train_size:train_size+valid_size]
    # # Creating a sampler for random sampling
    # train_sampler = SubsetRandomSampler(train_indices)
    # valid_sampler = SubsetRandomSampler(valid_indices)
    # # Creating data loaders with the samplers
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    # valid_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # print("train_loader: ", len(train_loader))
    # print("valid_loader: ", len(valid_loader))
    # print("test_loader: ", len(test_loader))

    return train_loader, valid_loader, None


def calculate_metrics(y_true, y_pred):
    """Calculate metrics including precision, recall, f1-score, and accuracy."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy


def threshold_predictions(y_probs, threshold=0.5):
    """Threshold probabilities to get binary predictions."""
    return (y_probs > threshold).astype(int)


def train_or_test(model, loader, device, criterion, optimizer=None, scheduler=None, mode="train", metric_files=None):
    """Generic function for training or testing the model with detailed metrics on progress bar and storing metrics to files."""
    # assert mode in ["train", "test"], "Mode must be 'train' or 'test'"
    if mode == "train":
        model.train()
    else:
        model.eval()
    
    epoch_loss = 0
    num_batches = len(loader)
    
    pbar = tqdm(total=num_batches, desc=f"{mode.capitalize()} Epoch", unit="batch")
    for batch_idx, data in enumerate(loader):
        DNAs, labels = data
        DNAs, labels = DNAs.to(torch.float32).to(device), labels.to(torch.float32).to(device)
        # print("DNAs.shape: ", DNAs.shape)
        # print("labels.shape: ", labels.shape)   

        DNAs, labels = torch.permute(DNAs, (0, 2, 1)), torch.permute(labels, (0, 2, 1))
        # print("DNAs.shape: ", DNAs.shape)
        # print("labels.shape: ", labels.shape)   
        if mode == "train":
            optimizer.zero_grad() # 
        with torch.set_grad_enabled(mode == "train"):
            loss, yp = model_fn(DNAs, labels, model, criterion)
            if mode == "train":
                loss.backward(), optimizer.step(), scheduler.step() 
        epoch_loss += loss.item()
        is_expr = (labels.sum(axis=(1,2)) >= 1).cpu().numpy()
        if np.any(is_expr):  # Process metrics only if there are expressions
            metrics = {}
            metrics[f"loss"] = loss.item() 
            for role, idx in [("neither", 0), ("donor", 1), ("acceptor", 2)]:
                # print("role: ", role, "\tidx: ", idx)
                y_true = labels[is_expr, idx, :].flatten().cpu().detach().numpy()
                y_pred = threshold_predictions(yp[is_expr, idx, :].flatten().cpu().detach().numpy())
                # print("y_true: ", y_true)
                # print("y_pred: ", y_pred)
                metrics[f"{role}_precision"], metrics[f"{role}_recall"], metrics[f"{role}_f1"], metrics[f"{role}_accuracy"] = calculate_metrics(y_true, y_pred)
                # Write metrics to files
                for name, value in metrics.items():
                    if metric_files and name in metric_files:
                        with open(metric_files[name], 'a') as f:
                            f.write(f"{value}\n")
            # Update progress bar
            # pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})
            print_dict = {}
            for k, v in metrics.items():
                if k not in ['donor_accuracy', 'acceptor_accuracy', 'neither_accuracy', 'neither_precision', 'neither_recall', 'neither_f1']:
                    print_dict[k] = f"{v:.4f}"
            pbar.set_postfix(print_dict)
        pbar.update(1)
    pbar.close()
    if mode == "train":
        # Additional logic for training
        pass






# def train_and_validate(model_m, sequence_length, project_root, SL, CL, N_GPUS, BATCH_SIZE):
#     """
#     Train and validate the model.
#     """
#     h5f_dir = f"{project_root}results/gene_sequences_and_labels/"
#     h5f = h5py.File(f'{h5f_dir}dataset_train.h5', 'r')
#     num_idx = len(list(h5f.keys()))//2
#     idx_all = np.random.permutation(num_idx)
#     idx_train = idx_all[:int(0.9*num_idx)]
#     idx_valid = idx_all[int(0.9*num_idx):]#int(0.2*num_idx)]
#     print("Number of training datasets: ", len(idx_train))
#     print("Number of validation datasets: ", len(idx_valid))
#     EPOCH_NUM = 10*len(idx_train)
#     experiment = f"SpliceAI_{SL}chunk_{sequence_length}flank_MANE_exp"
#     output_dir = f'{project_root}results/{experiment}/{sys.argv[2]}/'
#     os.makedirs(output_dir, exist_ok=True)
#     # files = {name: open(f'{output_dir}{name}_results.txt', 'w') for name in ['training', 'training_loss', 'validation', 'validation_loss']}
    
#     start_time = time.time()
#     for epoch_num in range(EPOCH_NUM):
#         pass
#         print("Epoch number: ", epoch_num)
#         idx = np.random.choice(idx_train)
#         X = h5f['X' + str(idx)][:]
#         Y = h5f['Y' + str(idx)][:]
#         # print("X.shape: ", X.shape)
#         # print("Y.shape: ", len(Y[0]))
#         Xc, Yc = clip_datapoints(X, Y, CL, 2) 
#         print("Xc.shape: ", Xc)
#         print("Yc.shape: ", Yc[0])

#     #     # unique, counts = np.unique(Xc, return_counts=True)
#     #     # unique, counts = np.unique([Yc], return_counts=True)
#     #     # print("unique: ", unique)
#     #     # print("counts: ", counts)
#     #     # print("Xc.shape: ", Xc.shape)
#     #     # print("Yc.shape: ", len(Yc[0]))
#     #     history = model_m.fit(Xc, Yc, batch_size=BATCH_SIZE, verbose=0)
#     #     # # NEW: Capture the loss value from the last batch of the current epoch
#     #     # current_loss = history.history['loss'][-1]  # Assuming 'loss' is the key for training loss
#     #     # # NEW: Write the current epoch number and loss to the training results file
#     #     # training_loss_results_file.write(f'{current_loss}\n')
#     #     # # training_results_file.flush()  # Ensure the written content is saved to the file
#     #     if (epoch_num+1) % len(idx_train) == 0:
#     #         print("--------------------------------------------------------------")
#     #         ########################################
#     #         # Validation set metrics
#     #         ########################################
#     #         print("\n\033[1mValidation set metrics:\033[0m")
#     #         Y_true_1 = [[] for t in range(1)]
#     #         Y_true_2 = [[] for t in range(1)]
#     #         Y_pred_1 = [[] for t in range(1)]
#     #         Y_pred_2 = [[] for t in range(1)]
#     #         for idx in idx_valid:
#     #             X = h5f['X' + str(idx)][:]
#     #             Y = h5f['Y' + str(idx)][:]
#     #             Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS)
#     #             Yp = model_m.predict(Xc, batch_size=BATCH_SIZE)
#     #             # After predicting with the validation set
#     #             # val_loss, val_metrics = model_m.evaluate(Xc, Yc, batch_size=BATCH_SIZE, verbose=0)
#     #             val_loss = model_m.evaluate(Xc, Yc, batch_size=BATCH_SIZE, verbose=0)
#     #             print(f"val_loss: {val_loss}")
#     #             files["validation_loss"].write(f'{val_loss}\n')
#     #             if not isinstance(Yp, list):
#     #                 Yp = [Yp]
#     #             for t in range(1):
#     #                 is_expr = (Yc[t].sum(axis=(1,2)) >= 1)
#     #                 Y_true_1[t].extend(Yc[t][is_expr, :, 1].flatten())
#     #                 Y_true_2[t].extend(Yc[t][is_expr, :, 2].flatten())
#     #                 Y_pred_1[t].extend(Yp[t][is_expr, :, 1].flatten())
#     #                 Y_pred_2[t].extend(Yp[t][is_expr, :, 2].flatten())
#     #         print("epoch_num: ", epoch_num)
#     #         print("\n\033[1mAcceptor:\033[0m")
#     #         for t in range(1):
#     #             print_topl_statistics(np.asarray(Y_true_1[t]),
#     #                                 np.asarray(Y_pred_1[t]), files["validation"], type='acceptor')
#     #         print("\n\033[1mDonor:\033[0m")
#     #         for t in range(1):
#     #             print_topl_statistics(np.asarray(Y_true_2[t]),
#     #                                 np.asarray(Y_pred_2[t]), files["validation"], type='donor')
#     #         ########################################
#     #         # Training set metrics
#     #         ########################################
#     #         print("\n\033[1mTraining set metrics:\033[0m")
#     #         Y_true_1 = [[] for t in range(1)]
#     #         Y_true_2 = [[] for t in range(1)]
#     #         Y_pred_1 = [[] for t in range(1)]
#     #         Y_pred_2 = [[] for t in range(1)]
#     #         for idx in idx_train[:len(idx_valid)]:
#     #             X = h5f['X' + str(idx)][:]
#     #             Y = h5f['Y' + str(idx)][:]
#     #             Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS)
#     #             Yp = model_m.predict(Xc, batch_size=BATCH_SIZE)
#     #             # After predicting with the training set
#     #             train_loss = model_m.evaluate(Xc, Yc, batch_size=BATCH_SIZE, verbose=0)
#     #             print(f"train_loss: {train_loss}")
#     #             files["training_loss"].write(f'{train_loss}\n')
#     #             if not isinstance(Yp, list):
#     #                 Yp = [Yp]
#     #             for t in range(1):
#     #                 is_expr = (Yc[t].sum(axis=(1,2)) >= 1)
#     #                 Y_true_1[t].extend(Yc[t][is_expr, :, 1].flatten())
#     #                 Y_true_2[t].extend(Yc[t][is_expr, :, 2].flatten())
#     #                 Y_pred_1[t].extend(Yp[t][is_expr, :, 1].flatten())
#     #                 Y_pred_2[t].extend(Yp[t][is_expr, :, 2].flatten())
#     #         print("\n\033[1mAcceptor:\033[0m")
#     #         for t in range(1):
#     #             print_topl_statistics(np.asarray(Y_true_1[t]),
#     #                                 np.asarray(Y_pred_1[t]), files["training"], type='acceptor')
#     #         print("\n\033[1mDonor:\033[0m")
#     #         for t in range(1):
#     #             print_topl_statistics(np.asarray(Y_true_2[t]),
#     #                                 np.asarray(Y_pred_2[t]), files["training"], type='donor')
#     #         print("Learning rate: %.5f" % (kb.get_value(model_m.optimizer.lr)))
#     #         print("--- %s seconds ---" % (time.time() - start_time))
#     #         start_time = time.time()
#     #         print("--------------------------------------------------------------")
#     #         model_m.save(f'{output_dir}/Models/SpliceAI' + sys.argv[1]
#     #                 + '_c' + '_' + experiment + '.h5')
#     #         if (epoch_num+1) >= 6*len(idx_train):
#     #             # Learning rate decay
#     #             kb.set_value(model_m.optimizer.lr,
#     #                         0.5*kb.get_value(model_m.optimizer.lr))
#     # for file in files.values():
#     #     file.close()
#     print("--- %s seconds ---" % (time.time() - start_time))
#     print("--------------------------------------------------------------")
#     h5f.close()





def main():
    chunk_size = sys.argv[1]
    flanking_size = sys.argv[2]
    device = setup_device()
    # target = "RefSeq_gene_sequences_and_labels"
    target = "gene_sequences_and_labels"
    data_dir, model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths(chunk_size, flanking_size)
    print("* data_dir: ", data_dir)
    print("* model_output_base: ", model_output_base)
    print("* log_output_train_base: ", log_output_train_base)
    print("* log_output_val_base: ", log_output_val_base)
    print("* log_output_test_base: ", log_output_test_base)
    train_loader, valid_loader, test_loader = load_data(data_dir, model_output_base, chunk_size, flanking_size)
    model, criterion, optimizer, scheduler = initialize_model_and_optim(device, len(train_loader), flanking_size)

    train_metric_files = {
        'neither_precision': f'{log_output_train_base}/neither_precision.txt',
        'neither_recall': f'{log_output_train_base}/neither_recall.txt',
        'neither_f1': f'{log_output_train_base}/neither_f1.txt',
        'neither_accuracy': f'{log_output_train_base}/neither_accuracy.txt',
        'donor_precision': f'{log_output_train_base}/donor_precision.txt',
        'donor_recall': f'{log_output_train_base}/donor_recall.txt',
        'donor_f1': f'{log_output_train_base}/donor_f1.txt',
        'donor_accuracy': f'{log_output_train_base}/donor_accuracy.txt',
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
        'donor_precision': f'{log_output_val_base}/donor_precision.txt',
        'donor_recall': f'{log_output_val_base}/donor_recall.txt',
        'donor_f1': f'{log_output_val_base}/donor_f1.txt',
        'donor_accuracy': f'{log_output_val_base}/donor_accuracy.txt',
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
        'donor_precision': f'{log_output_test_base}/donor_precision.txt',
        'donor_recall': f'{log_output_test_base}/donor_recall.txt',
        'donor_f1': f'{log_output_test_base}/donor_f1.txt',
        'donor_accuracy': f'{log_output_test_base}/donor_accuracy.txt',
        'acceptor_precision': f'{log_output_test_base}/acceptor_precision.txt',
        'acceptor_recall': f'{log_output_test_base}/acceptor_recall.txt',
        'acceptor_f1': f'{log_output_test_base}/acceptor_f1.txt',
        'acceptor_accuracy': f'{log_output_test_base}/acceptor_accuracy.txt',
        'loss': f'{log_output_test_base}/loss.txt'
    }

    for epoch in range(EPOCH_NUM):
        train_or_test(model, train_loader, device, criterion, optimizer, scheduler, mode="train", metric_files=train_metric_files)
        train_or_test(model, valid_loader, device, criterion, optimizer, scheduler, mode="valid", metric_files=valid_metric_files)
        torch.save(model, f'{model_output_base}/splan_'+str(epoch)+'.pt')

    # # Storing validation set.
    # train_or_test(model, test_loader, device, criterion, optimizer, scheduler, mode="test", metric_files=test_metric_files)

    # chunk_size = sys.argv[1]
    # flanking_size = sys.argv[2]
    # device = setup_device()
    # # target = "RefSeq_gene_sequences_and_labels"
    # target = "gene_sequences_and_labels"
    # data_dir, model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths(target)
    # print("* data_dir: ", data_dir)
    # print("* model_output_base: ", model_output_base)
    # print("* log_output_train_base: ", log_output_train_base)
    # print("* log_output_val_base: ", log_output_val_base)
    # print("* log_output_test_base: ", log_output_test_base)

    # project_root = "/Users/chaokuan-hao/Documents/Projects/spliceAI-MANE/"
    # h5f_dir = f"{project_root}results/gene_sequences_and_labels/"
    # h5f = h5py.File(f'{h5f_dir}dataset_train.h5', 'r')
    # num_idx = len(list(h5f.keys()))//2
    # idx_all = np.random.permutation(num_idx)
    # idx_train = idx_all[:int(0.9*num_idx)]
    # idx_valid = idx_all[int(0.9*num_idx):]#int(0.2*num_idx)]
    # print("Number of training datasets: ", len(idx_train))
    # print("Number of validation datasets: ", len(idx_valid))
    # EPOCH_NUM = 10*len(idx_train)
    # experiment = f"SpliceAI_{SL}chunk_{flanking_size}flank_MANE_exp"
    # output_dir = f'{project_root}results/{experiment}/{sys.argv[2]}/'
    # os.makedirs(output_dir, exist_ok=True)
    # # files = {name: open(f'{output_dir}{name}_results.txt', 'w') for name in ['training', 'training_loss', 'validation', 'validation_loss']}
    
    start_time = time.time()
    # N_GPUS = 2
    # for epoch_num in range(EPOCH_NUM):
    #     # pass
    #     # print("Epoch number: ", epoch_num)
    #     idx = np.random.choice(idx_train)
    #     X = h5f['X' + str(idx)][:]
    #     Y = h5f['Y' + str(idx)][:]

    #     # X.shape:  (640, 15000, 4)
    #     # Y.shape:  640
    #     print("X.shape: ", X.shape)
    #     print("Y.shape: ", len(Y[0]))

    #     # Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS) 
    #     # print("Xc.shape: ", Xc)
    #     # print("Yc.shape: ", Yc[0])















        # history = model_m.fit(Xc, Yc, batch_size=BATCH_SIZE, verbose=0)
    #     # # NEW: Capture the loss value from the last batch of the current epoch
    #     # current_loss = history.history['loss'][-1]  # Assuming 'loss' is the key for training loss
    #     # # NEW: Write the current epoch number and loss to the training results file
    #     # training_loss_results_file.write(f'{current_loss}\n')
    #     # # training_results_file.flush()  # Ensure the written content is saved to the file
    #     if (epoch_num+1) % len(idx_train) == 0:
    #         print("--------------------------------------------------------------")
    #         ########################################
    #         # Validation set metrics
    #         ########################################
    #         print("\n\033[1mValidation set metrics:\033[0m")
    #         Y_true_1 = [[] for t in range(1)]
    #         Y_true_2 = [[] for t in range(1)]
    #         Y_pred_1 = [[] for t in range(1)]
    #         Y_pred_2 = [[] for t in range(1)]
    #         for idx in idx_valid:
    #             X = h5f['X' + str(idx)][:]
    #             Y = h5f['Y' + str(idx)][:]
    #             Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS)
    #             Yp = model_m.predict(Xc, batch_size=BATCH_SIZE)
    #             # After predicting with the validation set
    #             # val_loss, val_metrics = model_m.evaluate(Xc, Yc, batch_size=BATCH_SIZE, verbose=0)
    #             val_loss = model_m.evaluate(Xc, Yc, batch_size=BATCH_SIZE, verbose=0)
    #             print(f"val_loss: {val_loss}")
    #             files["validation_loss"].write(f'{val_loss}\n')
    #             if not isinstance(Yp, list):
    #                 Yp = [Yp]
    #             for t in range(1):
    #                 is_expr = (Yc[t].sum(axis=(1,2)) >= 1)
    #                 Y_true_1[t].extend(Yc[t][is_expr, :, 1].flatten())
    #                 Y_true_2[t].extend(Yc[t][is_expr, :, 2].flatten())
    #                 Y_pred_1[t].extend(Yp[t][is_expr, :, 1].flatten())
    #                 Y_pred_2[t].extend(Yp[t][is_expr, :, 2].flatten())
    #         print("epoch_num: ", epoch_num)
    #         print("\n\033[1mAcceptor:\033[0m")
    #         for t in range(1):
    #             print_topl_statistics(np.asarray(Y_true_1[t]),
    #                                 np.asarray(Y_pred_1[t]), files["validation"], type='acceptor')
    #         print("\n\033[1mDonor:\033[0m")
    #         for t in range(1):
    #             print_topl_statistics(np.asarray(Y_true_2[t]),
    #                                 np.asarray(Y_pred_2[t]), files["validation"], type='donor')
    #         ########################################
    #         # Training set metrics
    #         ########################################
    #         print("\n\033[1mTraining set metrics:\033[0m")
    #         Y_true_1 = [[] for t in range(1)]
    #         Y_true_2 = [[] for t in range(1)]
    #         Y_pred_1 = [[] for t in range(1)]
    #         Y_pred_2 = [[] for t in range(1)]
    #         for idx in idx_train[:len(idx_valid)]:
    #             X = h5f['X' + str(idx)][:]
    #             Y = h5f['Y' + str(idx)][:]
    #             Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS)
    #             Yp = model_m.predict(Xc, batch_size=BATCH_SIZE)
    #             # After predicting with the training set
    #             train_loss = model_m.evaluate(Xc, Yc, batch_size=BATCH_SIZE, verbose=0)
    #             print(f"train_loss: {train_loss}")
    #             files["training_loss"].write(f'{train_loss}\n')
    #             if not isinstance(Yp, list):
    #                 Yp = [Yp]
    #             for t in range(1):
    #                 is_expr = (Yc[t].sum(axis=(1,2)) >= 1)
    #                 Y_true_1[t].extend(Yc[t][is_expr, :, 1].flatten())
    #                 Y_true_2[t].extend(Yc[t][is_expr, :, 2].flatten())
    #                 Y_pred_1[t].extend(Yp[t][is_expr, :, 1].flatten())
    #                 Y_pred_2[t].extend(Yp[t][is_expr, :, 2].flatten())
    #         print("\n\033[1mAcceptor:\033[0m")
    #         for t in range(1):
    #             print_topl_statistics(np.asarray(Y_true_1[t]),
    #                                 np.asarray(Y_pred_1[t]), files["training"], type='acceptor')
    #         print("\n\033[1mDonor:\033[0m")
    #         for t in range(1):
    #             print_topl_statistics(np.asarray(Y_true_2[t]),
    #                                 np.asarray(Y_pred_2[t]), files["training"], type='donor')
    #         print("Learning rate: %.5f" % (kb.get_value(model_m.optimizer.lr)))
    #         print("--- %s seconds ---" % (time.time() - start_time))
    #         start_time = time.time()
    #         print("--------------------------------------------------------------")
    #         model_m.save(f'{output_dir}/Models/SpliceAI' + sys.argv[1]
    #                 + '_c' + '_' + experiment + '.h5')
    #         if (epoch_num+1) >= 6*len(idx_train):
    #             # Learning rate decay
    #             kb.set_value(model_m.optimizer.lr,
    #                         0.5*kb.get_value(model_m.optimizer.lr))
    # for file in files.values():
    #     file.close()
    print("--- %s seconds ---" % (time.time() - start_time))
    print("--------------------------------------------------------------")
    # h5f.close()



    # train_loader, valid_loader, test_loader = load_data(data_dir, chunk_size, flanking_size)
    # model, criterion, optimizer, scheduler = initialize_model_and_optim(device, len(train_loader), flanking_size)

    # train_metric_files = {
    #     'neither_precision': f'{log_output_train_base}/neither_precision.txt',
    #     'neither_recall': f'{log_output_train_base}/neither_recall.txt',
    #     'neither_f1': f'{log_output_train_base}/neither_f1.txt',
    #     'neither_accuracy': f'{log_output_train_base}/neither_accuracy.txt',
    #     'donor_precision': f'{log_output_train_base}/donor_precision.txt',
    #     'donor_recall': f'{log_output_train_base}/donor_recall.txt',
    #     'donor_f1': f'{log_output_train_base}/donor_f1.txt',
    #     'donor_accuracy': f'{log_output_train_base}/donor_accuracy.txt',
    #     'acceptor_precision': f'{log_output_train_base}/acceptor_precision.txt',
    #     'acceptor_recall': f'{log_output_train_base}/acceptor_recall.txt',
    #     'acceptor_f1': f'{log_output_train_base}/acceptor_f1.txt',
    #     'acceptor_accuracy': f'{log_output_train_base}/acceptor_accuracy.txt',
    #     'loss': f'{log_output_train_base}/loss.txt'
    # }
    # valid_metric_files = {
    #     'neither_precision': f'{log_output_val_base}/neither_precision.txt',
    #     'neither_recall': f'{log_output_val_base}/neither_recall.txt',
    #     'neither_f1': f'{log_output_val_base}/neither_f1.txt',
    #     'neither_accuracy': f'{log_output_val_base}/neither_accuracy.txt',
    #     'donor_precision': f'{log_output_val_base}/donor_precision.txt',
    #     'donor_recall': f'{log_output_val_base}/donor_recall.txt',
    #     'donor_f1': f'{log_output_val_base}/donor_f1.txt',
    #     'donor_accuracy': f'{log_output_val_base}/donor_accuracy.txt',
    #     'acceptor_precision': f'{log_output_val_base}/acceptor_precision.txt',
    #     'acceptor_recall': f'{log_output_val_base}/acceptor_recall.txt',
    #     'acceptor_f1': f'{log_output_val_base}/acceptor_f1.txt',
    #     'acceptor_accuracy': f'{log_output_val_base}/acceptor_accuracy.txt',
    #     'loss': f'{log_output_val_base}/loss.txt'
    # }
    # test_metric_files = {
    #     'neither_precision': f'{log_output_test_base}/neither_precision.txt',
    #     'neither_recall': f'{log_output_test_base}/neither_recall.txt',
    #     'neither_f1': f'{log_output_test_base}/neither_f1.txt',
    #     'neither_accuracy': f'{log_output_test_base}/neither_accuracy.txt',
    #     'donor_precision': f'{log_output_test_base}/donor_precision.txt',
    #     'donor_recall': f'{log_output_test_base}/donor_recall.txt',
    #     'donor_f1': f'{log_output_test_base}/donor_f1.txt',
    #     'donor_accuracy': f'{log_output_test_base}/donor_accuracy.txt',
    #     'acceptor_precision': f'{log_output_test_base}/acceptor_precision.txt',
    #     'acceptor_recall': f'{log_output_test_base}/acceptor_recall.txt',
    #     'acceptor_f1': f'{log_output_test_base}/acceptor_f1.txt',
    #     'acceptor_accuracy': f'{log_output_test_base}/acceptor_accuracy.txt',
    #     'loss': f'{log_output_test_base}/loss.txt'
    # }

    # for epoch in range(EPOCH_NUM):
    #     train_or_test(model, train_loader, device, criterion, optimizer, scheduler, mode="train", metric_files=train_metric_files)
    #     train_or_test(model, valid_loader, device, criterion, optimizer, scheduler, mode="valid", metric_files=valid_metric_files)
    #     torch.save(model, f'{model_output_base}/splan_'+str(epoch)+'.pt')

    # # Storing validation set.
    # train_or_test(model, test_loader, device, criterion, optimizer, scheduler, mode="test", metric_files=test_metric_files)
if __name__ == "__main__":
    main()
