import numpy as np
JUNC_START = 200
JUNC_END = 600

CL_MAX = 10000
SL = 800

#############################
# Global variable definition
#############################
EPOCH_NUM = 15
BATCH_SIZE = 200
N_WORKERS = 1
L = 64
JUNC_THRESHOLD = 0.1

# W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
#                 11, 11, 11, 11, 21, 21, 21, 21,
#                 21, 21, 21, 21])
# AR = np.asarray([1, 1, 1, 1, 5, 5, 5, 5,
#                  10, 10, 10, 10, 15, 15, 15, 15,
#                 20, 20, 20, 20])

# CL = 2 * np.sum(AR*(W-1))

TRAIN_RATIO = 0.8
