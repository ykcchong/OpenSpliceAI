# Usage: python Step_4_verify_h5_file.py <train, test, all>

# import necessary libraries
import h5py
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
from constants import *
import time 

# record start time for benchmark
start_time = time.time()

assert sys.argv[1] in ['train', 'test', 'all']

# construct the filename and open h5 file
filename = data_dir + 'dataset' + '_' + sys.argv[1] + '.h5'

with h5py.File(filename, 'r') as hf:
    # print the available dataset keys in the file
    print(f"Dataset keys: {list(hf.keys())}\n\n")

    # convert datasets to PyTorch tensors and display their shapes
    X0_tensor = torch.from_numpy(hf['X0'][:]).float()  
    Y0_tensor = torch.from_numpy(hf['Y0'][:]) 
    print(f"X0 shape: {X0_tensor.shape}, Y0 shape: {Y0_tensor.shape}")

    # process a specific dataset ('X3') for visualization
    x = torch.from_numpy(hf['X3'][:]).float()  
    y = torch.from_numpy(hf['Y3'][:])  
    print(f"x[0].shape: {x[0].shape}, y[0].shape: {y[0].shape}")

    # plot the sum of the last entry in the 'X3' dataset along its rows
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 15000])
    ax.plot(x[-1].sum(axis=1))  # Sum over rows and plot

    # additional info
    print("x[0].sum(axis=1): ", len(x[len(x)-1].sum(axis=1))) # Length of row sum in the last 'X3' entry
    print("x[0].sum(axis=0): ", len(x[len(x)-1].sum(axis=0))) # Length of column sum in the last 'X3' entry
    print(f"(x[0].sum(axis=1) == 0).sum(): {(x[0].sum(axis=1) == 0).sum()}") # Number of zero-sum rows in the first 'X3' entry

# END
print(("--- %s seconds ---" % (time.time() - start_time)))

plt.show()