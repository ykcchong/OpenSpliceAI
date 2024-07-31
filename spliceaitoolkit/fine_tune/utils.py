###############################################################################
'''This code has functions which process the information in the .h5 files
datafile_{}_{}.h5 and convert them into a format usable by Keras.'''
###############################################################################

import numpy as np
import torch
from math import ceil
from sklearn.metrics import average_precision_score
from spliceaitoolkit.constants import *

# assert CL_max % 2 == 0

def ceil_div(x, y):
    return int(ceil(float(x)/y))

def clip_datapoints(X, Y, CL, CL_max, N_GPUS):
    # This function is necessary to make sure of the following:
    # (i) Each time model_m.fit is called, the number of datapoints is a
    # multiple of N_GPUS. Failure to ensure this often results in crashes.
    # (ii) If the required context length is less than CL_max, then
    # appropriate clipping is done below.
    # Additionally, Y is also converted to a list (the .h5 files store 
    # them as an array).

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
    return - torch.mean(y_true[:, 0, :]*torch.log(y_pred[:, 0, :]+1e-10)
                        + y_true[:, 1, :]*torch.log(y_pred[:, 1, :]+1e-10)
                        + y_true[:, 2, :]*torch.log(y_pred[:, 2, :]+1e-10))

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, weights=None):
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
    weights = 1000
    return - torch.mean(y_true[:, 0, :]*torch.log(y_pred[:, 0, :]+epsilon) * torch.pow(torch.sub(1, y_pred[:, 0, :]), gamma)
                        + y_true[:, 1, :]*torch.log(y_pred[:, 1, :]+epsilon) * torch.pow(torch.sub(1, y_pred[:, 1, :]), gamma)
                        + y_true[:, 2, :]*torch.log(y_pred[:, 2, :]+epsilon) * torch.pow(torch.sub(1, y_pred[:, 2, :]), gamma))


