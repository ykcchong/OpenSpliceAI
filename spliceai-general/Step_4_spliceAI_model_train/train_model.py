import os
import sys
import time
import numpy as np
import h5py
import tensorflow.keras.backend as kb
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from spliceai import SpliceAI, categorical_crossentropy_2d  # Assuming this is a custom model
from utils import clip_datapoints, print_topl_statistics
from multi_gpu import make_parallel


def get_hyperparameters(sequence_length, N_GPUS):
    """
    Returns the hyperparameters based on the input sequence length.
    """
    if sequence_length == 80:
        W = np.asarray([11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1])
        BATCH_SIZE = 18*N_GPUS
    elif sequence_length == 400:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
        BATCH_SIZE = 18*N_GPUS
    elif sequence_length == 2000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10])
        BATCH_SIZE = 12*N_GPUS
    elif sequence_length == 10000:
        W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21, 41, 41, 41, 41])
        AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10, 25, 25, 25, 25])
        BATCH_SIZE = 6*N_GPUS
    else:
        raise ValueError("Invalid sequence length.")
    return {'W': W, 'AR': AR, 'BATCH_SIZE': BATCH_SIZE}


def setup_environment():
    """
    Set up the environment by configuring paths and ensuring necessary directories are in sys.path.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)


def validate_sequence_length(sequence_length):
    """
    Ensure the sequence length is one of the predefined values.
    """
    assert sequence_length in [80, 400, 2000, 10000], "Sequence length must be one of [80, 400, 2000, 10000]."


def initialize_model(L, W, AR, N_GPUS):
    """
    Initialize the SpliceAI model.
    """
    model = SpliceAI(L, W, AR)
    model.summary()
    model_m = make_parallel(model, N_GPUS)
    model_m.compile(loss=categorical_crossentropy_2d, optimizer=Adam())
    return model_m


def train_and_validate(model_m, sequence_length, project_root, SL, CL, N_GPUS, BATCH_SIZE):
    """
    Train and validate the model.
    """
    h5f_dir = f"{project_root}results/gene_sequences_and_labels/"
    h5f = h5py.File(f'{h5f_dir}dataset_train.h5', 'r')
    num_idx = len(list(h5f.keys()))//2
    idx_all = np.random.permutation(num_idx)
    idx_train = idx_all[:int(0.9*num_idx)]
    idx_valid = idx_all[int(0.9*num_idx):]#int(0.2*num_idx)]
    print("Number of training datasets: ", len(idx_train))
    print("Number of validation datasets: ", len(idx_valid))
    EPOCH_NUM = 10*len(idx_train)
    experiment = f"SpliceAI_{SL}chunk_{sequence_length}flank_MANE_exp"
    output_dir = f'{project_root}results/{experiment}/{sys.argv[2]}/'
    os.makedirs(output_dir, exist_ok=True)
    files = {name: open(f'{output_dir}{name}_results.txt', 'w') for name in ['training', 'training_loss', 'validation', 'validation_loss']}
    start_time = time.time()
    for epoch_num in range(EPOCH_NUM):
        print("Epoch number: ", epoch_num)
        idx = np.random.choice(idx_train)
        X = h5f['X' + str(idx)][:]
        Y = h5f['Y' + str(idx)][:]
        # print("X.shape: ", X.shape)
        # print("Y.shape: ", len(Y[0]))
        Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS) 
        # print("Xc.shape: ", Xc)
        # print("Yc.shape: ", Yc[0])

        # unique, counts = np.unique(Xc, return_counts=True)
        # unique, counts = np.unique([Yc], return_counts=True)
        # print("unique: ", unique)
        # print("counts: ", counts)
        # print("Xc.shape: ", Xc.shape)
        # print("Yc.shape: ", len(Yc[0]))
        history = model_m.fit(Xc, Yc, batch_size=BATCH_SIZE, verbose=0)
        # # NEW: Capture the loss value from the last batch of the current epoch
        # current_loss = history.history['loss'][-1]  # Assuming 'loss' is the key for training loss
        # # NEW: Write the current epoch number and loss to the training results file
        # training_loss_results_file.write(f'{current_loss}\n')
        # # training_results_file.flush()  # Ensure the written content is saved to the file
        if (epoch_num+1) % len(idx_train) == 0:
            print("--------------------------------------------------------------")
            ########################################
            # Validation set metrics
            ########################################
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
                print(f"val_loss: {val_loss}")
                files["validation_loss"].write(f'{val_loss}\n')
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
                                    np.asarray(Y_pred_1[t]), files["validation"], type='acceptor')
            print("\n\033[1mDonor:\033[0m")
            for t in range(1):
                print_topl_statistics(np.asarray(Y_true_2[t]),
                                    np.asarray(Y_pred_2[t]), files["validation"], type='donor')
            ########################################
            # Training set metrics
            ########################################
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
                print(f"train_loss: {train_loss}")
                files["training_loss"].write(f'{train_loss}\n')
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
                                    np.asarray(Y_pred_1[t]), files["training"], type='acceptor')
            print("\n\033[1mDonor:\033[0m")
            for t in range(1):
                print_topl_statistics(np.asarray(Y_true_2[t]),
                                    np.asarray(Y_pred_2[t]), files["training"], type='donor')
            print("Learning rate: %.5f" % (kb.get_value(model_m.optimizer.lr)))
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            print("--------------------------------------------------------------")
            model_m.save(f'{output_dir}/Models/SpliceAI' + sys.argv[1]
                    + '_c' + '_' + experiment + '.h5')
            if (epoch_num+1) >= 6*len(idx_train):
                # Learning rate decay
                kb.set_value(model_m.optimizer.lr,
                            0.5*kb.get_value(model_m.optimizer.lr))
    for file in files.values():
        file.close()
    h5f.close()


def main():
    setup_environment()
    sequence_length = int(sys.argv[1])
    validate_sequence_length(sequence_length)

    from constants import SL, CL_max  # Assuming these are defined in constants.py

    N_GPUS = 2  # Number of GPUs
    hyperparameters = get_hyperparameters(sequence_length, N_GPUS)
    W, AR, BATCH_SIZE = hyperparameters['W'], hyperparameters['AR'], hyperparameters['BATCH_SIZE']
    CL = 2 * np.sum(AR * (W - 1))
    assert CL <= CL_max and CL == sequence_length

    print(f"\033[1mContext nucleotides: {CL}\033[0m")
    print(f"\033[1mSequence length (output): {SL}\033[0m")

    model_m = initialize_model(32, W, AR, N_GPUS)  # 32 is the number of convolution kernels (L)

    project_root = "/Users/chaokuan-hao/Documents/Projects/spliceAI-MANE/"
    train_and_validate(model_m, sequence_length, project_root, SL, CL, N_GPUS, BATCH_SIZE)

if __name__ == "__main__":
    main()






# # Get the absolute path of the current script's directory
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # Get the parent directory
# parent_dir = os.path.dirname(current_dir)
# # Append the parent directory to sys.path
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)

# from constants import * 

# assert sequence_length in [80, 400, 2000, 10000]

# ###############################################################################
# # Model parameter definition
# ###############################################################################
# L = 32
# N_GPUS = 2

# # Hyper-parameters:
# # L: Number of convolution kernels
# # W: Convolution window size in each residual unit
# # AR: Atrous rate in each residual unit
# if sequence_length == 80:
#     W = np.asarray([11, 11, 11, 11])
#     AR = np.asarray([1, 1, 1, 1])
#     BATCH_SIZE = 18*N_GPUS
# elif sequence_length == 400:
#     W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11])
#     AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4])
#     BATCH_SIZE = 18*N_GPUS
# elif sequence_length == 2000:
#     W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
#                     21, 21, 21, 21])
#     AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
#                      10, 10, 10, 10])
#     BATCH_SIZE = 12*N_GPUS
# elif sequence_length == 10000:
#     W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
#                     21, 21, 21, 21, 41, 41, 41, 41])
#     AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
#                      10, 10, 10, 10, 25, 25, 25, 25])
#     BATCH_SIZE = 6*N_GPUS
# CL = 2 * np.sum(AR*(W-1))
# assert CL <= CL_max and CL == sequence_length
# print("\033[1mContext nucleotides: %d\033[0m" % (CL))
# print("\033[1mSequence length (output): %d\033[0m" % (SL))

# model = SpliceAI(L, W, AR)
# model.summary()
# model_m = make_parallel(model, N_GPUS)
# model_m.compile(loss=categorical_crossentropy_2d, optimizer='adam')


# ###############################################################################
# # Training and validation
# ###############################################################################
# # h5f = h5py.File(data_dir + 'dataset_test_0.h5', 'r')
# project_root = "/Users/chaokuan-hao/Documents/Projects/spliceAI-MANE/"
# output_dir = f"{project_root}results/gene_sequences_and_labels/"
# h5f = h5py.File(f'{output_dir}dataset_train.h5', 'r')
# num_idx = len(list(h5f.keys()))//2
# idx_all = np.random.permutation(num_idx)
# idx_train = idx_all[:int(0.9*num_idx)]
# idx_valid = idx_all[int(0.9*num_idx):]
# print("Number of training datasets: ", len(idx_train))
# print("Number of validation datasets: ", len(idx_valid))
# EPOCH_NUM = 15*len(idx_train)
# start_time = time.time()
# # experiment = f"Splan_MultiHeadAttention_500chunk_{sys.argv[1]}flank_isoforms_exp"
# # experiment = "SpliceAI_500chunk_2000flank"
# experiment = f"SpliceAI_{SL}chunk_{sys.argv[1]}flank_MANE_exp"

# output_dir = f'{project_root}results/{experiment}/'
# os.makedirs(output_dir, exist_ok=True)
# training_results_fn = f'{output_dir}training_results.txt'
# training_loss_results_fn = f'{output_dir}training_loss_results.txt'
# validation_results_fn = f'{output_dir}validation_results.txt'
# validation_loss_results_fn = f'{output_dir}validation_loss_results.txt'

# training_results_file = open(training_results_fn, 'w')
# training_loss_results_file = open(training_loss_results_fn, 'w')
# validation_results_file = open(validation_results_fn, 'w')
# validation_loss_results_file = open(validation_loss_results_fn, 'w')


# for epoch_num in range(EPOCH_NUM):
#     idx = np.random.choice(idx_train)
#     X = h5f['X' + str(idx)][:]
#     Y = h5f['Y' + str(idx)][:]
#     # print("X.shape: ", X.shape)
#     # print("Y.shape: ", len(Y[0]))
#     Xc, Yc = clip_datapoints(X, Y, CL, N_GPUS) 
#     print("Xc.shape: ", Xc.shape)
#     print("Yc.shape: ", len(Yc[0]))
#     history = model_m.fit(Xc, Yc, batch_size=BATCH_SIZE, verbose=0)

#     # # NEW: Capture the loss value from the last batch of the current epoch
#     # current_loss = history.history['loss'][-1]  # Assuming 'loss' is the key for training loss
#     # # NEW: Write the current epoch number and loss to the training results file
#     # training_loss_results_file.write(f'{current_loss}\n')
#     # # training_results_file.flush()  # Ensure the written content is saved to the file
#     if (epoch_num+1) % len(idx_train) == 0:
#         # Printing metrics (see utils.py for details)

#         print("--------------------------------------------------------------")
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
#             validation_loss_results_file.write(f'{val_loss}\n')

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
#                                   np.asarray(Y_pred_1[t]), validation_results_file, type='acceptor')

#         print("\n\033[1mDonor:\033[0m")
#         for t in range(1):
#             print_topl_statistics(np.asarray(Y_true_2[t]),
#                                   np.asarray(Y_pred_2[t]), validation_results_file, type='donor')

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
#             training_loss_results_file.write(f'{train_loss}\n')

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
#                                   np.asarray(Y_pred_1[t]), training_results_file, type='acceptor')

#         print("\n\033[1mDonor:\033[0m")
#         for t in range(1):
#             print_topl_statistics(np.asarray(Y_true_2[t]),
#                                   np.asarray(Y_pred_2[t]), training_results_file, type='donor')

#         print("Learning rate: %.5f" % (kb.get_value(model_m.optimizer.lr)))
#         print("--- %s seconds ---" % (time.time() - start_time))
#         start_time = time.time()

#         print("--------------------------------------------------------------")

#         model.save('./Models/SpliceAI' + sys.argv[1]
#                    + '_c' + sys.argv[2] + '_' + experiment + '.h5')

#         if (epoch_num+1) >= 6*len(idx_train):
#             kb.set_value(model_m.optimizer.lr,
#                          0.5*kb.get_value(model_m.optimizer.lr))
#             # Learning rate decay

# training_results_file.close()
# training_loss_results_file.close()
# validation_results_file.close()
# validation_loss_results_file.close()
# h5f.close()
# ###############################################################################

