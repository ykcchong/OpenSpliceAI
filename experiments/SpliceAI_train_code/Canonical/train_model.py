###############################################################################
# This file contains the code to train the SpliceAI model.
###############################################################################

import numpy as np
import sys
import time
import h5py
import os
import keras.backend as kb
import tensorflow as tf
from spliceai import *
from utils import *
from multi_gpu import *
from constants import *
import wandb
import argparse

###############################################################################
# Getting parameters from the command line
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--disable-wandb', '-d', action='store_true', default=False)
parser.add_argument('--output', '-p', type=str)
parser.add_argument('--project-name', '-s', type=str)
parser.add_argument('--flanking-size', '-f', type=int, default=80)
parser.add_argument('--exp-num', '-e', type=str, default=0)
parser.add_argument('--training-target', '-t', type=str, default="SpliceAI")
parser.add_argument('--train-dataset', '-train', type=str)
parser.add_argument('--test-dataset', '-test', type=str)
args = parser.parse_args()

disable_wandb = args.disable_wandb
output = args.output
project_name = args.project_name
flanking_size = args.flanking_size
exp_num = args.exp_num
training_target = args.training_target
train_dataset = args.train_dataset
test_dataset = args.test_dataset
output_dir = f'{output}{project_name}/'
os.makedirs(output_dir, exist_ok=True)

if disable_wandb:
    os.environ['WANDB_MODE'] = 'disabled'
wandb.init(project=f'{project_name}_{training_target}_{SL}_{flanking_size}_{exp_num}', reinit=True)
###############################################################################
# End of getting parameters from the command line
###############################################################################

L = 32
N_GPUS = 2
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
# Hyper-parameters:
# L: Number of convolution kernels
# W: Convolution window size in each residual unit
# AR: Atrous rate in each residual unit

CL = 2 * np.sum(AR*(W-1))
# assert CL <= CL_max and CL == int(flanking_size)
print ("\033[1mContext nucleotides: %d\033[0m" % (CL))
print ("\033[1mSequence length (output): %d\033[0m" % (SL))
model = SpliceAI(L, W, AR)
model.summary()
# model = make_parallel(model, N_GPUS)
model.compile(loss=categorical_crossentropy_2d, optimizer='adam')

print(f'output: {output}')
print(f'project_name: {project_name}')
print(f'flanking_size: {flanking_size}')
print(f'exp_num: {exp_num}')
print(f'training_target: {training_target}')
print(f'train_dataset: {train_dataset}')
print(f'test_dataset: {test_dataset}')
###############################################################################
# Training and validation
###############################################################################
h5f = h5py.File(train_dataset, 'r')
# h5f = h5py.File(test_dataset, 'r')

num_idx = len(h5f.keys())//2
idx_all = np.random.permutation(num_idx)
idx_train = idx_all[:int(0.3*num_idx)]
idx_valid = idx_all[int(0.3*num_idx):int(0.4*num_idx)]
# idx_train = idx_all[:int(0.2*num_idx)]
# idx_valid = idx_all[int(0.3*num_idx):int(0.4*num_idx)]
print(f'idx_all: {idx_all}')
print(f'idx_train: {idx_train}')
print(f'idx_valid: {idx_valid}')

EPOCH_NUM = 10*len(idx_train)
start_time = time.time()
d_training_results_fn = f'{output_dir}training_donor_results.txt'
a_training_results_fn = f'{output_dir}training_acceptor_results.txt'
training_loss_results_fn = f'{output_dir}training_loss_results.txt'
d_validation_results_fn = f'{output_dir}validation_donor_results.txt'
a_validation_results_fn = f'{output_dir}validation_acceptor_results.txt'
validation_loss_results_fn = f'{output_dir}validation_loss_results.txt'

d_training_results_file = open(d_training_results_fn, 'w')
a_training_results_file = open(a_training_results_fn, 'w')
training_loss_results_file = open(training_loss_results_fn, 'w')
d_validation_results_file = open(d_validation_results_fn, 'w')
a_validation_results_file = open(a_validation_results_fn, 'w')
validation_loss_results_file = open(validation_loss_results_fn, 'w')

for epoch_num in range(EPOCH_NUM):
    idx = np.random.choice(idx_train)
    X = h5f['X' + str(idx)][:]
    Y = h5f['Y' + str(idx)][:]
    Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS) 
    history = model.fit(Xc, Yc, batch_size=BATCH_SIZE, verbose=0)
    loss = history.history['loss'][0]  # Get the loss for the current epoch
    print(f"Epoch {epoch_num+1}/{EPOCH_NUM}, Loss: {loss}")

    if (epoch_num+1) % len(idx_train) == 0:
        ############################################
        # Validation set metrics
        ############################################
        print ("--------------------------------------------------------------")
        print ("\n\033[1mValidation set metrics:\033[0m")
        Y_true_1 = [[] for t in range(1)]
        Y_true_2 = [[] for t in range(1)]
        Y_pred_1 = [[] for t in range(1)]
        Y_pred_2 = [[] for t in range(1)]
        val_running_loss = 0
        for idx in idx_valid:
            X = h5f['X' + str(idx)][:]
            Y = h5f['Y' + str(idx)][:]
            Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS)
            Yp = model.predict(Xc, batch_size=BATCH_SIZE)
            val_loss = model.evaluate(Xc, Yc, batch_size=BATCH_SIZE, verbose=0)
            print("val_loss: ", val_loss)
            val_running_loss += val_loss
            if not isinstance(Yp, list):
                Yp = [Yp]
            for t in range(1):
                is_expr = (Yc[t].sum(axis=(1,2)) >= 1)
                Y_true_1[t].extend(Yc[t][is_expr, :, 1].flatten())
                Y_true_2[t].extend(Yc[t][is_expr, :, 2].flatten())
                Y_pred_1[t].extend(Yp[t][is_expr, :, 1].flatten())
                Y_pred_2[t].extend(Yp[t][is_expr, :, 2].flatten())
        print("epoch_num: ", epoch_num)
        validation_loss_results_file.write(f'{val_running_loss / len(idx_valid)}\n')
        print("\n\033[1mAcceptor:\033[0m")
        for t in range(1):
            acceptor_topkl_accuracy, acceptor_auprc = print_topl_statistics(np.asarray(Y_true_1[t]), np.asarray(Y_pred_1[t]), a_validation_results_file, type='acceptor')
        print("\n\033[1mDonor:\033[0m")
        for t in range(1):
            donor_topkl_accuracy, donor_auprc = print_topl_statistics(np.asarray(Y_true_2[t]), np.asarray(Y_pred_2[t]), d_validation_results_file, type='donor')
        wandb.log({
            f'validation/loss': val_running_loss / len(idx_valid),
            f'validation/topk_acceptor': acceptor_topkl_accuracy,
            f'validation/topk_donor': donor_topkl_accuracy,
            f'validation/auprc_acceptor': acceptor_auprc,
            f'validation/auprc_donor': donor_auprc,
        })

        ############################################
        # Training set metrics
        ############################################
        print ("\n\033[1mTraining set metrics:\033[0m")
        Y_true_1 = [[] for t in range(1)]
        Y_true_2 = [[] for t in range(1)]
        Y_pred_1 = [[] for t in range(1)]
        Y_pred_2 = [[] for t in range(1)]
        train_running_loss = 0
        for idx in idx_train[:len(idx_valid)]:
            X = h5f['X' + str(idx)][:]
            Y = h5f['Y' + str(idx)][:]
            Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS)
            Yp = model.predict(Xc, batch_size=BATCH_SIZE)
            train_loss = model.evaluate(Xc, Yc, batch_size=BATCH_SIZE, verbose=0)
            print("train_loss: ", train_loss)
            train_running_loss += train_loss
            if not isinstance(Yp, list):
                Yp = [Yp]
            for t in range(1):
                is_expr = (Yc[t].sum(axis=(1,2)) >= 1)
                Y_true_1[t].extend(Yc[t][is_expr, :, 1].flatten())
                Y_true_2[t].extend(Yc[t][is_expr, :, 2].flatten())
                Y_pred_1[t].extend(Yp[t][is_expr, :, 1].flatten())
                Y_pred_2[t].extend(Yp[t][is_expr, :, 2].flatten())
        training_loss_results_file.write(f'{train_running_loss/len(idx_train[:len(idx_valid)])}\n')
        print("\n\033[1mAcceptor:\033[0m")
        for t in range(1):
            acceptor_topkl_accuracy, acceptor_auprc = print_topl_statistics(np.asarray(Y_true_1[t]), np.asarray(Y_pred_1[t]), a_training_results_file, type='acceptor')
        print("\n\033[1mDonor:\033[0m")
        for t in range(1):
            donor_topkl_accuracy, donor_auprc = print_topl_statistics(np.asarray(Y_true_2[t]), np.asarray(Y_pred_2[t]), d_training_results_file, type='donor')
        wandb.log({
            f'training/loss': train_running_loss/len(idx_train[:len(idx_valid)]),
            f'training/topk_acceptor': acceptor_topkl_accuracy,
            f'training/topk_donor': donor_topkl_accuracy,
            f'training/auprc_acceptor': acceptor_auprc,
            f'training/auprc_donor': donor_auprc,
        })

        # Learning rate decay
        print ("Learning rate: %.5f" % (kb.get_value(model.optimizer.lr)))
        print ("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        print ("--------------------------------------------------------------")
        model.save(f'{output_dir}Models/SpliceAI_' + str(flanking_size)
                   + '_g' + str(exp_num) + '_' + epoch_num +'.h5')
        if (epoch_num+1) >= 6*len(idx_train):
            kb.set_value(model.optimizer.lr,
                         0.5*kb.get_value(model.optimizer.lr))
a_training_results_file.close()
d_training_results_file.close()
training_loss_results_file.close()
a_validation_results_file.close()
d_validation_results_file.close()
validation_loss_results_file.close()
h5f.close()
# ###############################################################################