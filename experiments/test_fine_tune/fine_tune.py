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
    device_str = "cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" else "cpu"
    return torch.device(device_str)


def initialize_paths(output_dir, project_name, flanking_size, exp_num, sequence_length, loss_fun, random_seed):
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


def initialize_model_and_optim(device, flanking_size, model_path, num_layers_to_unfreeze):
    L = 32
    N_GPUS = 2
    W = np.asarray([11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1])
    BATCH_SIZE = 18 * N_GPUS
    if int(flanking_size) == 80:
        W = np.asarray([11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1])
        BATCH_SIZE = 18 * N_GPUS
    elif int(flanking_size) == 400:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
        BATCH_SIZE = 18 * N_GPUS
    elif int(flanking_size) == 2000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10])
        BATCH_SIZE = 12 * N_GPUS
    elif int(flanking_size) == 10000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21, 41, 41, 41, 41])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10, 25, 25, 25, 25])
        BATCH_SIZE = 6 * N_GPUS
    
    CL = 2 * np.sum(AR * (W - 1))
    print("\033[1mContext nucleotides: %d\033[0m" % (CL))
    print("\033[1mSequence length (output): %d\033[0m" % (SL))
    model = SpliceAI(L, W, AR).to(device)

    # Load the pretrained model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last `num_layers_to_unfreeze` layers
    if num_layers_to_unfreeze > 0:
        layers = list(model.children())[-num_layers_to_unfreeze:]  # Assuming the model layers are accessible like this
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True

    # Set up optimizer and scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    print(model, file=sys.stderr)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'N_GPUS': N_GPUS}
    return model, None, optimizer, scheduler, params


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


def load_data_from_shard(h5f, shard_idx, device, batch_size, params, shuffle=False):
    X = h5f[f'X{shard_idx}'][:].transpose(0, 2, 1)
    Y = h5f[f'Y{shard_idx}'][0, ...].transpose(0, 2, 1)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    ds = TensorDataset(X, Y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, pin_memory=True)


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
            DNAs, labels = clip_datapoints(DNAs, labels, params["CL"], flanking_size, params["N_GPUS"])
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
            DNAs, labels = clip_datapoints(DNAs, labels, params["CL"], flanking_size, params["N_GPUS"])
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


def fine_tune(args):

    print('Running OpenSpliceAI with fine-tune mode.')

    output_dir = args.output_dir
    project_name = args.project_name
    sequence_length = SL
    flanking_size = int(args.flanking_size)
    exp_num = args.exp_num
    model_path = args.model_path
    num_layers_to_unfreeze = args.num_layers_to_unfreeze
    assert int(flanking_size) in [80, 400, 2000, 10000]
    if args.disable_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    
    wandb.init(project=f'{project_name}', reinit=True)
    device = setup_device()
    print("device: ", device, file=sys.stderr)
    model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths(output_dir, project_name, flanking_size, exp_num, sequence_length, args.loss, args.random_seed)
    print("* Project name: ", args.project_name, file=sys.stderr)
    print("* Model_output_base: ", model_output_base, file=sys.stderr)
    print("* Log_output_train_base: ", log_output_train_base, file=sys.stderr)
    print("* Log_output_val_base: ", log_output_val_base, file=sys.stderr)
    print("* Log_output_test_base: ", log_output_test_base, file=sys.stderr)

    training_dataset = args.train_dataset
    testing_dataset = args.test_dataset

    print("Training_dataset: ", training_dataset, file=sys.stderr)
    print("Testing_dataset: ", testing_dataset, file=sys.stderr)
    print("Model architecture: ", model_path, file=sys.stderr)
    print("Loss function: ", args.loss, file=sys.stderr)
    print("Flanking sequence size: ", args.flanking_size, file=sys.stderr)
    print("Exp number: ", args.exp_num, file=sys.stderr)

    train_h5f = h5py.File(training_dataset, 'r')
    test_h5f = h5py.File(testing_dataset, 'r')
    batch_num = len(train_h5f.keys()) // 2
    RANDOM_SEED = args.random_seed

    print("Batch_num: ", batch_num, file=sys.stderr)
    print("RANDOM_SEED: ", RANDOM_SEED, file=sys.stderr)
    np.random.seed(RANDOM_SEED)

    idxs = np.random.permutation(batch_num)
    train_idxs = idxs[:int(0.9 * batch_num)]
    val_idxs = idxs[int(0.9 * batch_num):]
    test_idxs = np.arange(len(test_h5f.keys()) // 2)

    print("train_idxs: ", train_idxs, file=sys.stderr)
    print("val_idxs: ", val_idxs, file=sys.stderr)
    print("test_idxs: ", test_idxs, file=sys.stderr)

    model, criterion, optimizer, scheduler, params = initialize_model_and_optim(device, flanking_size, model_path, num_layers_to_unfreeze)
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

    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_patience = 5
    for epoch in range(20):
        print("\n--------------------------------------------------------------")
        current_lr = optimizer.param_groups[0]['lr']
        print(f">> Epoch {epoch + 1}; Current Learning Rate: {current_lr}")
        wandb.log({
            f'train/learning_rate': current_lr,
        })
        start_time = time.time()
        train_loss = train_epoch(model, train_h5f, train_idxs, params["BATCH_SIZE"], args.loss, optimizer, scheduler, device, params, train_metric_files, flanking_size, run_mode="train")
        val_loss = valid_epoch(model, train_h5f, val_idxs, params["BATCH_SIZE"], args.loss, device, params, valid_metric_files, flanking_size, run_mode="validation")
        test_loss = valid_epoch(model, test_h5f, test_idxs, params["BATCH_SIZE"], args.loss, device, params, test_metric_files, flanking_size, run_mode="test")
        
        scheduler.step(val_loss)
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(model.state_dict(), f"{model_output_base}/model_{epoch}.pt")
            print("New best model saved.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epochs.")
            if epochs_no_improve >= n_patience:
                print("Early stopping triggered.")
                break
        print("--- %s seconds ---" % (time.time() - start_time))
        print("--------------------------------------------------------------")
    train_h5f.close()
    test_h5f.close()


if __name__ == "__main__":
    parser_fine_tune = argparse.ArgumentParser(description="Fine-tune the SpliceAI model on new data")
    parser_fine_tune.add_argument("--output_dir", '-o', type=str, required=True, help="Output directory for model checkpoints and logs")
    parser_fine_tune.add_argument("--project_name", '-s', type=str, required=True, help="Project name for organizing outputs")
    parser_fine_tune.add_argument("--exp_num", '-e', type=int, default=0, help="Experiment number")
    parser_fine_tune.add_argument("--flanking_size", '-f', type=int, default=80, choices=[80, 400, 2000, 10000], help="Flanking sequence size")
    parser_fine_tune.add_argument("--model_path", '-m', type=str, required=True, help="Model architecture to use")
    parser_fine_tune.add_argument("--loss", '-l', type=str, default='cross_entropy_loss', choices=["cross_entropy_loss", "focal_loss"], help="Loss function for training")
    parser_fine_tune.add_argument("--train_dataset", '-train', type=str, required=True, help="Path to the training dataset")
    parser_fine_tune.add_argument("--test_dataset", '-test', type=str, required=True, help="Path to the testing dataset")
    parser_fine_tune.add_argument("--disable_wandb", '-d', action='store_true', default=False, help="Disable Weights & Biases logging")
    parser_fine_tune.add_argument("--random_seed", '-r', type=int, default=42, help="Random seed for reproducibility")
    parser_fine_tune.add_argument("--unfreeze", '-u', type=int, default=1, help="Number of layers to unfreeze for fine-tuning")
    args = parser_fine_tune.parse_args()
    fine_tune(args)
