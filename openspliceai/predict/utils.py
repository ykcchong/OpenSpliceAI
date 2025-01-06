###############################################################################
'''This code has functions which process the information in the .h5 files
datafile_{}_{}.h5 and convert them into a format usable by Keras.'''
###############################################################################

from math import ceil
from openspliceai.constants import *

def ceil_div(x, y):
    """
    Calculate the ceiling of a division between two numbers.

    Parameters:
    - x (int): Numerator
    - y (int): Denominator

    Returns:
    - int: The ceiling of the division result.
    """
    return int(ceil(float(x)/y))

# FOR TESTING PURPOSES
import psutil
def log_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / (1024 * 1024)} MB", file=sys.stderr)

# SETUP INITIALIZATION
import os
def initialize_constants(flanking_size, hdf_threshold_len=0, flush_predict_threshold=500, chunk_size=100, split_fasta_threshold=1500000):
    
    assert int(flanking_size) in [80, 400, 2000, 10000]
    
    global CL_max                  # context length for sequence prediction (flanking size sum)
    global HDF_THRESHOLD_LEN       # maximum size before reading sequence into an HDF file for storage
    global FLUSH_PREDICT_THRESHOLD # maximum number of predictions before flushing to file
    global CHUNK_SIZE              # chunk size for loading hdf5 dataset
    global SPLIT_FASTA_THRESHOLD   # maximum length of fasta entry before splitting
    
    CL_max = flanking_size
    HDF_THRESHOLD_LEN = hdf_threshold_len
    FLUSH_PREDICT_THRESHOLD = flush_predict_threshold
    CHUNK_SIZE = chunk_size
    SPLIT_FASTA_THRESHOLD = split_fasta_threshold
    
def initialize_paths(output_dir, flanking_size, sequence_length, model_arch='SpliceAI'):
    """Initialize project directories and create them if they don't exist."""

    BASENAME = f"{model_arch}_{sequence_length}_{flanking_size}"
    model_pred_outdir = f"{output_dir}/{BASENAME}/"
    os.makedirs(model_pred_outdir, exist_ok=True)

    return model_pred_outdir