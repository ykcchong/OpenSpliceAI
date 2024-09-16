import argparse
import h5py
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import platform
from openspliceai.train_base.spliceai import *
from openspliceai.train_base.utils import *
from openspliceai.constants import *
import h5py
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import wandb


def initialize_model_and_optim(device, flanking_size, pretrained_model, unfreeze, unfreeze_all):
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
    # Initialize the model
    model = SpliceAI(L, W, AR).to(device)
    # Print the shapes of the parameters in the initialized model
    print("\nInitialized model parameter shapes:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}", end=", ")

    # Load the pretrained model
    state_dict = torch.load(pretrained_model, map_location=device)

    # Filter out unnecessary keys and load matching keys into model
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

    # Load state dict into the model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Print missing and unexpected keys
    print("\nMissing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    print("\n unfreeze_all:", unfreeze_all)
    if not unfreeze_all:
        # Freeze all layers first
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze the last `unfreeze` layers
        if unfreeze > 0:
            # Unfreeze the last few layers (example: last residual unit)
            for param in model.residual_units[-unfreeze].parameters():
                param.requires_grad = True
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    print(model, file=sys.stderr)    
    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'N_GPUS': N_GPUS}
    return model, optimizer, scheduler, params


def fine_tune(args):
    print('Running OpenSpliceAI with fine-tune mode.')
    # assert training_target in ["RefSeq", "MANE", "SpliceAI", "SpliceAI27"]
    device = setup_environment(args)
    model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths(args)
    train_h5f, test_h5f, batch_num = load_datasets(args)
    train_idxs, val_idxs, test_idxs = generate_indices(batch_num, args.random_seed, test_h5f)
    model, optimizer, scheduler, params = initialize_model_and_optim(device, args.flanking_size, args.pretrained_model, args.unfreeze, args.unfreeze_all)
    
    params["RANDOM_SEED"] = args.random_seed
    train_metric_files = create_metric_files(log_output_train_base)
    valid_metric_files = create_metric_files(log_output_val_base)
    test_metric_files = create_metric_files(log_output_test_base)
    train_model(model, optimizer, scheduler, train_h5f, test_h5f, train_idxs, val_idxs, test_idxs, 
                model_output_base, args, device, params, train_metric_files, valid_metric_files, test_metric_files)
    train_h5f.close()
    test_h5f.close()



    # output_dir = args.output_dir
    # project_name = args.project_name
    # sequence_length = SL
    # flanking_size = int(args.flanking_size)
    # exp_num = args.exp_num
    # pretrained_model = args.pretrained_model
    # unfreeze = args.unfreeze
    # assert int(flanking_size) in [80, 400, 2000, 10000]
    # if not args.enable_wandb:
    #     os.environ['WANDB_MODE'] = 'disabled'
    
    # wandb.init(project=f'{project_name}', reinit=True)
    # device = setup_device()
    # print("device: ", device, file=sys.stderr)
    # model_output_base, log_output_train_base, log_output_val_base, log_output_test_base = initialize_paths(output_dir, project_name, flanking_size, exp_num, sequence_length, args.loss, args.random_seed)
    # print("* Project name: ", args.project_name, file=sys.stderr)
    # print("* Model_output_base: ", model_output_base, file=sys.stderr)
    # print("* Log_output_train_base: ", log_output_train_base, file=sys.stderr)
    # print("* Log_output_val_base: ", log_output_val_base, file=sys.stderr)
    # print("* Log_output_test_base: ", log_output_test_base, file=sys.stderr)

    # training_dataset = args.train_dataset
    # testing_dataset = args.test_dataset

    # print("* Training_dataset: ", training_dataset, file=sys.stderr)
    # print("* Testing_dataset: ", testing_dataset, file=sys.stderr)
    # print("* Model architecture: ", pretrained_model, file=sys.stderr)
    # print("* Loss function: ", args.loss, file=sys.stderr)
    # print("* Flanking sequence size: ", args.flanking_size, file=sys.stderr)
    # print("* Exp number: ", args.exp_num, file=sys.stderr)

    # train_h5f = h5py.File(training_dataset, 'r')
    # test_h5f = h5py.File(testing_dataset, 'r')
    # batch_num = len(train_h5f.keys()) // 2
    # RANDOM_SEED = args.random_seed

    # print("* Batch_num: ", batch_num, file=sys.stderr)
    # print("* RANDOM_SEED: ", RANDOM_SEED, file=sys.stderr)
    # print("***************************************\n\n", file=sys.stderr)
    # np.random.seed(RANDOM_SEED)

    # idxs = np.random.permutation(batch_num)
    # train_idxs = idxs[:int(0.9 * batch_num)]
    # val_idxs = idxs[int(0.9 * batch_num):]
    # test_idxs = np.arange(len(test_h5f.keys()) // 2)

    # # print("train_idxs: ", train_idxs, file=sys.stderr)
    # # print("val_idxs: ", val_idxs, file=sys.stderr)
    # # print("test_idxs: ", test_idxs, file=sys.stderr)

    # model, criterion, optimizer, scheduler, params = initialize_model_and_optim(device, flanking_size, pretrained_model, unfreeze, args.unfreeze_all)
    # params["RANDOM_SEED"] = RANDOM_SEED
    # train_metric_files = {
    #     'donor_topk_all': f'{log_output_train_base}/donor_topk_all.txt',
    #     'donor_topk': f'{log_output_train_base}/donor_topk.txt',
    #     'donor_auprc': f'{log_output_train_base}/donor_auprc.txt',
    #     'donor_accuracy': f'{log_output_train_base}/donor_accuracy.txt',
    #     'donor_precision': f'{log_output_train_base}/donor_precision.txt',
    #     'donor_recall': f'{log_output_train_base}/donor_recall.txt',
    #     'donor_f1': f'{log_output_train_base}/donor_f1.txt',
    #     'acceptor_topk_all': f'{log_output_train_base}/acceptor_topk_all.txt',
    #     'acceptor_topk': f'{log_output_train_base}/acceptor_topk.txt',
    #     'acceptor_auprc': f'{log_output_train_base}/acceptor_auprc.txt',
    #     'acceptor_accuracy': f'{log_output_train_base}/acceptor_accuracy.txt',
    #     'acceptor_precision': f'{log_output_train_base}/acceptor_precision.txt',
    #     'acceptor_recall': f'{log_output_train_base}/acceptor_recall.txt',
    #     'acceptor_f1': f'{log_output_train_base}/acceptor_f1.txt',
    #     'accuracy': f'{log_output_train_base}/accuracy.txt',
    #     'loss_batch': f'{log_output_train_base}/loss_batch.txt',
    #     'loss_every_update': f'{log_output_train_base}/loss_every_update.txt',
    # }
    # valid_metric_files = {
    #     'donor_topk_all': f'{log_output_val_base}/donor_topk_all.txt',
    #     'donor_topk': f'{log_output_val_base}/donor_topk.txt',
    #     'donor_auprc': f'{log_output_val_base}/donor_auprc.txt',
    #     'donor_accuracy': f'{log_output_val_base}/donor_accuracy.txt',
    #     'donor_precision': f'{log_output_val_base}/donor_precision.txt',
    #     'donor_recall': f'{log_output_val_base}/donor_recall.txt',
    #     'donor_f1': f'{log_output_val_base}/donor_f1.txt',
    #     'acceptor_topk_all': f'{log_output_val_base}/acceptor_topk_all.txt',
    #     'acceptor_topk': f'{log_output_val_base}/acceptor_topk.txt',
    #     'acceptor_auprc': f'{log_output_val_base}/acceptor_auprc.txt',
    #     'acceptor_accuracy': f'{log_output_val_base}/acceptor_accuracy.txt',
    #     'acceptor_precision': f'{log_output_val_base}/acceptor_precision.txt',
    #     'acceptor_recall': f'{log_output_val_base}/acceptor_recall.txt',
    #     'acceptor_f1': f'{log_output_val_base}/acceptor_f1.txt',
    #     'accuracy': f'{log_output_val_base}/accuracy.txt',
    #     'loss_batch': f'{log_output_val_base}/loss_batch.txt',
    #     'loss_every_update': f'{log_output_val_base}/loss_every_update.txt',
    # }
    # test_metric_files = {
    #     'donor_topk_all': f'{log_output_test_base}/donor_topk_all.txt',
    #     'donor_topk': f'{log_output_test_base}/donor_topk.txt',
    #     'donor_auprc': f'{log_output_test_base}/donor_auprc.txt',
    #     'donor_accuracy': f'{log_output_test_base}/donor_accuracy.txt',
    #     'donor_precision': f'{log_output_test_base}/donor_precision.txt',
    #     'donor_recall': f'{log_output_test_base}/donor_recall.txt',
    #     'donor_f1': f'{log_output_test_base}/donor_f1.txt',
    #     'acceptor_topk_all': f'{log_output_test_base}/acceptor_topk_all.txt',
    #     'acceptor_topk': f'{log_output_test_base}/acceptor_topk.txt',
    #     'acceptor_auprc': f'{log_output_test_base}/acceptor_auprc.txt',
    #     'acceptor_accuracy': f'{log_output_test_base}/acceptor_accuracy.txt',
    #     'acceptor_precision': f'{log_output_test_base}/acceptor_precision.txt',
    #     'acceptor_recall': f'{log_output_test_base}/acceptor_recall.txt',
    #     'acceptor_f1': f'{log_output_test_base}/acceptor_f1.txt',      
    #     'accuracy': f'{log_output_test_base}/accuracy.txt',
    #     'loss_batch': f'{log_output_test_base}/loss_batch.txt',
    #     'loss_every_update': f'{log_output_test_base}/loss_every_update.txt',
    # }

    #     # 'topk_donor': f'{log_output_train_base}/donor_topk.txt',
    #     # 'auprc_donor': f'{log_output_train_base}/donor_accuracy.txt',
    #     # 'topk_acceptor': f'{log_output_train_base}/acceptor_topk.txt',
    #     # 'auprc_acceptor': f'{log_output_train_base}/acceptor_accuracy.txt',
    #     # 'loss_batch': f'{log_output_train_base}/loss_batch.txt',
    #     # 'loss_every_update': f'{log_output_train_base}/loss_every_update.txt'


    # best_val_loss = float('inf')
    # epochs_no_improve = 0
    # for epoch in range(EPOCH_NUM):
    #     print("\n============================================================")
    #     current_lr = optimizer.param_groups[0]['lr']
    #     print(f">> Epoch {epoch + 1}; Current Learning Rate: {current_lr}")
    #     wandb.log({
    #         f'fine-tune/learning_rate': current_lr,
    #     })
    #     start_time = time.time()
    #     train_loss = train_epoch(model, train_h5f, train_idxs, params["BATCH_SIZE"], args.loss, optimizer, scheduler, device, params, train_metric_files, flanking_size, run_mode="train")
    #     val_loss = valid_epoch(model, train_h5f, val_idxs, params["BATCH_SIZE"], args.loss, device, params, valid_metric_files, flanking_size, run_mode="validation")
    #     test_loss = valid_epoch(model, test_h5f, test_idxs, params["BATCH_SIZE"], args.loss, device, params, test_metric_files, flanking_size, run_mode="test")

    #     print(f"Training Loss: {train_loss}")
    #     print(f"Validation Loss: {val_loss}")
    #     print(f"Testing Loss: {test_loss}")
        
    #     if args.early_stopping:
    #         scheduler.step(val_loss)
    #         if val_loss.item() < best_val_loss:
    #             best_val_loss = val_loss.item()
    #             torch.save(model.state_dict(), f"{model_output_base}/model_{epoch}.pt")
    #             print("New best model saved.")
    #             epochs_no_improve = 0
    #         else:
    #             epochs_no_improve += 1
    #             print(f"No improvement in validation loss for {epochs_no_improve} epochs.")
    #             if epochs_no_improve >= args.patience:
    #                 print("Early stopping triggered.")
    #                 break
    #     print("--- %s seconds ---" % (time.time() - start_time))
    #     print("============================================================")
    # train_h5f.close()
    # test_h5f.close()