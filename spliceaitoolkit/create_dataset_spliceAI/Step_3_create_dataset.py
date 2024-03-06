###############################################################################
'''This parser takes as input the .h5 file produced by create_datafile.py and
outputs a .h5 file with datapoints of the form (X, Y), which can be understood
by Keras models.'''
###############################################################################

# Usage: python Step_3_create_dataset.py <train, test, all> <0, 1, all>

# import necessary libraries
import h5py
import numpy as np
import sys
import time
from utils import *
from constants import *

# START
start_time = time.time()

assert sys.argv[1] in ['train', 'test', 'all']
assert sys.argv[2] in ['0', '1', 'all']

# open the source HDF5 file in read mode
with h5py.File(data_dir + 'datafile' + '_' + sys.argv[1] + '.h5', 'r') as h5f:

    # read datasets from the file
    SEQ = h5f['SEQ'][:]  # sequences
    STRAND = h5f['STRAND'][:]  # strand information
    TX_START = h5f['TX_START'][:]  # transcription start positions
    TX_END = h5f['TX_END'][:]  # transcription end positions
    JN_START = h5f['JN_START'][:]  # junction start positions
    JN_END = h5f['JN_END'][:]  # junction end positions

# open a new HDF5 file for writing processed data
with h5py.File(data_dir + 'dataset' + '_' + sys.argv[1] + '.h5', 'w') as h5f2:

    # define the size of each chunk to be processed
    CHUNK_SIZE = 100

    # print total number of sequences and how many chunks will be processed
    print("SEQ.shape[0]: ", SEQ.shape[0])
    print("SEQ.shape[0]//CHUNK_SIZE: ", SEQ.shape[0]//CHUNK_SIZE)

    # process data in chunks
    for i in range(SEQ.shape[0] // CHUNK_SIZE):
        # adjust CHUNK_SIZE for the last chunk if it contains fewer elements
        NEW_CHUNK_SIZE = CHUNK_SIZE + SEQ.shape[0] % CHUNK_SIZE if (i + 1) == SEQ.shape[0] // CHUNK_SIZE else CHUNK_SIZE
        print("NEW_CHUNK_SIZE: ", NEW_CHUNK_SIZE)

        X_batch = []  # initialize batch for input features
        Y_batch = [[] for _ in range(1)]  # initialize batch for labels, assuming a single label type for simplicity

        # process each sequence in the chunk
        for j in range(NEW_CHUNK_SIZE):
            idx = i * CHUNK_SIZE + j  # calculate the global index of the current sequence

            # decode the sequence data from bytes to strings
            SEQ_cvt = SEQ[idx].decode('utf-8')
            STRAND_cvt = STRAND[idx].decode('utf-8')
            TX_START_cvt = TX_START[idx].decode('utf-8')
            TX_END_cvt = TX_END[idx].decode('utf-8')
            JN_START_cvt = np.array([JN_START[idx][0].decode('utf-8')])
            JN_END_cvt = np.array([JN_END[idx][0].decode('utf-8')])

            # generate input features (X) and labels (Y) using the utility function
            X, Y = create_datapoints(SEQ_cvt, STRAND_cvt, TX_START_cvt, TX_END_cvt, JN_START_cvt, JN_END_cvt)

            # show shapes of generated matrices
            print("X.shape: ", X.shape)
            print("len(Y[0]): ", len(Y[0]))

            # append generated data to the batch
            X_batch.extend(X)
            for t in range(1):
                Y_batch[t].extend(Y[t])

        # convert batches to numpy arrays with appropriate data type
        X_batch = np.asarray(X_batch).astype('int8')
        for t in range(1):
            Y_batch[t] = np.asarray(Y_batch[t]).astype('int8')

        # show batch shapes
        print("X_batch.shape: ", X_batch.shape)
        print("len(Y_batch[0]): ", len(Y_batch[0]))

        # save the processed chunk to the new HDF5 file
        h5f2.create_dataset('X' + str(i), data=X_batch)
        h5f2.create_dataset('Y' + str(i), data=Y_batch)

# END  
print("--- %s seconds ---" % (time.time() - start_time))  
