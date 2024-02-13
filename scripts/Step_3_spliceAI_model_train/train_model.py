###############################################################################
# This file contains the code to train the SpliceAI model.
###############################################################################

import numpy as np
import sys
import time
import h5py
import tensorflow.keras.backend as kb
import tensorflow as tf
import os
from spliceai import *
from utils import *
from multi_gpu import *
from constants import * 

assert int(sys.argv[1]) in [80, 400, 2000, 10000]

###############################################################################
# Model parameter definition
###############################################################################
L = 32
N_GPUS = 2

if int(sys.argv[1]) == 80:
    W = np.asarray([11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1])
    BATCH_SIZE = 18*N_GPUS
elif int(sys.argv[1]) == 400:
    W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
    BATCH_SIZE = 18*N_GPUS
elif int(sys.argv[1]) == 2000:
    W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                    21, 21, 21, 21])
    AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                     10, 10, 10, 10])
    BATCH_SIZE = 12*N_GPUS
elif int(sys.argv[1]) == 10000:
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
assert CL <= CL_max and CL == int(sys.argv[1])
print("\033[1mContext nucleotides: %d\033[0m" % (CL))
print("\033[1mSequence length (output): %d\033[0m" % (SL))

model = SpliceAI(L, W, AR)
model.summary()
model_m = make_parallel(model, N_GPUS)
model_m.compile(loss=categorical_crossentropy_2d, optimizer='adam')

###############################################################################
# Training and validation
###############################################################################
# h5f = h5py.File(data_dir + 'dataset_test_0.h5', 'r')
project_root = "/Users/chaokuan-hao/Documents/Projects/spliceAI-MANE/"
output_dir = f"{project_root}results/gene_sequences_and_labels/"
h5f = h5py.File(f'{output_dir}dataset_train.h5', 'r')
num_idx = len(list(h5f.keys()))//2
idx_all = np.random.permutation(num_idx)
idx_train = idx_all[:int(0.9*num_idx)]
idx_valid = idx_all[int(0.9*num_idx):]
print("Number of training datasets: ", len(idx_train))
print("Number of validation datasets: ", len(idx_valid))
EPOCH_NUM = 15*len(idx_train)
start_time = time.time()
# experiment = f"Splan_MultiHeadAttention_500chunk_{sys.argv[1]}flank_isoforms_exp"
# experiment = "SpliceAI_500chunk_2000flank"
experiment = f"SpliceAI_500chunk_{sys.argv[1]}flank_MANE_exp"

output_dir = f'./results/{experiment}/'
os.makedirs(output_dir, exist_ok=True)
training_results_fn = f'{output_dir}training_results.txt'
training_loss_results_fn = f'{output_dir}training_loss_results.txt'
validation_results_fn = f'{output_dir}validation_results.txt'
validation_loss_results_fn = f'{output_dir}validation_loss_results.txt'

training_results_file = open(training_results_fn, 'w')
training_loss_results_file = open(training_loss_results_fn, 'w')
validation_results_file = open(validation_results_fn, 'w')
validation_loss_results_file = open(validation_loss_results_fn, 'w')


for epoch_num in range(EPOCH_NUM):
    idx = np.random.choice(idx_train)
    X = h5f['X' + str(idx)][:]
    Y = h5f['Y' + str(idx)][:]
    # print("X.shape: ", X.shape)
    # print("Y.shape: ", len(Y[0]))
    Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS) 
    print("Xc.shape: ", Xc.shape)
    print("Yc.shape: ", len(Yc[0]))
    history = model_m.fit(Xc, Yc, batch_size=BATCH_SIZE, verbose=0)

    # # NEW: Capture the loss value from the last batch of the current epoch
    # current_loss = history.history['loss'][-1]  # Assuming 'loss' is the key for training loss
    # # NEW: Write the current epoch number and loss to the training results file
    # training_loss_results_file.write(f'{current_loss}\n')
    # # training_results_file.flush()  # Ensure the written content is saved to the file
    if (epoch_num+1) % len(idx_train) == 0:
        # Printing metrics (see utils.py for details)

        print("--------------------------------------------------------------")
        print("\n\033[1mValidation set metrics:\033[0m")

        Y_true_1 = [[] for t in range(1)]
        Y_true_2 = [[] for t in range(1)]
        Y_pred_1 = [[] for t in range(1)]
        Y_pred_2 = [[] for t in range(1)]

        for idx in idx_valid:

            X = h5f['X' + str(idx)][:]
            Y = h5f['Y' + str(idx)][:]

            Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS)
            Yp = model_m.predict(Xc, batch_size=BATCH_SIZE)

            # After predicting with the validation set
            # val_loss, val_metrics = model_m.evaluate(Xc, Yc, batch_size=BATCH_SIZE, verbose=0)
            val_loss = model_m.evaluate(Xc, Yc, batch_size=BATCH_SIZE, verbose=0)
            validation_loss_results_file.write(f'{val_loss}\n')

            if not isinstance(Yp, list):
                Yp = [Yp]

            for t in range(1):

                is_expr = (Yc[t].sum(axis=(1,2)) >= 1)

                Y_true_1[t].extend(Yc[t][is_expr, :, 1].flatten())
                Y_true_2[t].extend(Yc[t][is_expr, :, 2].flatten())
                Y_pred_1[t].extend(Yp[t][is_expr, :, 1].flatten())
                Y_pred_2[t].extend(Yp[t][is_expr, :, 2].flatten())


        print("epoch_num: ", epoch_num)
        print("\n\033[1mAcceptor:\033[0m")
        for t in range(1):
            print_topl_statistics(np.asarray(Y_true_1[t]),
                                  np.asarray(Y_pred_1[t]), validation_results_file, type='acceptor')

        print("\n\033[1mDonor:\033[0m")
        for t in range(1):
            print_topl_statistics(np.asarray(Y_true_2[t]),
                                  np.asarray(Y_pred_2[t]), validation_results_file, type='donor')

        print("\n\033[1mTraining set metrics:\033[0m")

        Y_true_1 = [[] for t in range(1)]
        Y_true_2 = [[] for t in range(1)]
        Y_pred_1 = [[] for t in range(1)]
        Y_pred_2 = [[] for t in range(1)]

        for idx in idx_train[:len(idx_valid)]:

            X = h5f['X' + str(idx)][:]
            Y = h5f['Y' + str(idx)][:]

            Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS)
            Yp = model_m.predict(Xc, batch_size=BATCH_SIZE)
            
            # After predicting with the training set
            train_loss = model_m.evaluate(Xc, Yc, batch_size=BATCH_SIZE, verbose=0)
            training_loss_results_file.write(f'{train_loss}\n')

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
            print_topl_statistics(np.asarray(Y_true_1[t]),
                                  np.asarray(Y_pred_1[t]), training_results_file, type='acceptor')

        print("\n\033[1mDonor:\033[0m")
        for t in range(1):
            print_topl_statistics(np.asarray(Y_true_2[t]),
                                  np.asarray(Y_pred_2[t]), training_results_file, type='donor')

        print("Learning rate: %.5f" % (kb.get_value(model_m.optimizer.lr)))
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()

        print("--------------------------------------------------------------")

        model.save('./Models/SpliceAI' + sys.argv[1]
                   + '_c' + sys.argv[2] + '_' + experiment + '.h5')

        if (epoch_num+1) >= 6*len(idx_train):
            kb.set_value(model_m.optimizer.lr,
                         0.5*kb.get_value(model_m.optimizer.lr))
            # Learning rate decay

training_results_file.close()
training_loss_results_file.close()
validation_results_file.close()
validation_loss_results_file.close()

h5f.close()
        
###############################################################################

