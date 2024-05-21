import argparse
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import platform
import h5py
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from pyfaidx import Fasta
import wandb
from spliceaitoolkit.predict.spliceai import *
from spliceaitoolkit.predict.utils import *
from spliceaitoolkit.constants import *

# # FOR TESTING PURPOSES
# from spliceai import *
# from utils import *



RANDOM_SEED = 42 # for replicability
HDF_THRESHOLD_LEN = 5000 # maximum size before reading sequence into an HDF file for storage
CHUNK_SIZE = 100 # chunk size for loading hdf5 dataset

#####################
##      SETUP      ##
#####################

# initialize output directory for log files, predict.bed file
def initialize_paths(output_dir, flanking_size, sequence_length, model_arch='SpliceAI'):
    """Initialize project directories and create them if they don't exist."""

    BASENAME = f"{model_arch}_{sequence_length}_{flanking_size}"
    model_pred_outdir = f"{output_dir}/{BASENAME}/"

    log_output_base = f"{model_pred_outdir}LOG/"
    os.makedirs(log_output_base, exist_ok=True)

    return model_pred_outdir, log_output_base

def setup_device():
    """Select computation device based on availability."""
    device_str = "cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" else "cpu"
    return torch.device(device_str)

# load given model and get params
def load_model(device, flanking_size):
    """Loads the given model."""
    # Hyper-parameters:
    # L: Number of convolution kernels
    # W: Convolution window size in each residual unit
    # AR: Atrous rate in each residual unit
    L = 32
    W = np.asarray([11, 11, 11, 11])
    AR = np.asarray([1, 1, 1, 1])
    N_GPUS = 2
    BATCH_SIZE = 18*N_GPUS

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

    CL = 2 * np.sum(AR*(W-1))

    print("\033[1mContext nucleotides: %d\033[0m" % (CL))
    print("\033[1mSequence length (output): %d\033[0m" % (SL))
    
    model = SpliceAI(L, W, AR).to(device)
    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'N_GPUS': N_GPUS}

    print(model, file=sys.stderr)
    return model, params


#########################
##   DATA PROCESSING   ##
#########################
    
'''MODIFY TO JUST GET SEQUENCES, BUILD SEP FUNCTION THAT READS INTO TEMPFILE IF SIZE <=5K'''
def get_sequences(fasta_file, output_dir, neg_strands=None):
    """
    Extract sequences for each protein-coding gene, process them based on strand orientation,
    and save data in file depending on the sequence size (HDF file if sequence >HDF_THRESHOLD_LEN, 
    else temp file).

    Parameters:
    - fasta_file: Path to the FASTA file.
    - output_dir: Directory to save the output files.
    - neg_strands (list): List of IDs for fasta entries where the sequence is on the reverse strand. Default is None.

    Returns:
    - Path to datafile.
    """

    # detect sequence length, determine file saving method to use
    total_length = 0
    use_hdf = False

    # NOTE: always creates uppercase sequence, uses [1,0]-indexed sequences, takes simple name from FASTA
    genes = Fasta(fasta_file, one_based_attributes=True, read_long_names=False, sequence_always_upper=True) 

    for record in genes:
        total_length += len(genes[record.name])
        if total_length > HDF_THRESHOLD_LEN:
            use_hdf = True
            break

    NAME = [] # Gene Header
    SEQ  = [] # Sequences

    # obtain the headers and sequences from FASTA file
    for record in genes:
        record = record[len(record)] # seems to be pyfaidx quirk? 
        seq_id = record.fancy_name
        sequence = record.seq

        # reverse strand if explicitly specified
        if neg_strands is not None and record.name in neg_strands: 
            seq_id = str(seq_id) + ':-'
            sequence = sequence.reverse.complement
        else:
            seq_id = str(seq_id) + ':+'
        
        NAME.append(seq_id)
        SEQ.append(str(sequence))
    
    # write the sequences to datafile
    if use_hdf:
        datafile_path = f'{output_dir}datafile.h5'
        dt = h5py.string_dtype(encoding='utf-8')
        with h5py.File(datafile_path, 'w') as datafile: # hdf5 information file
            datafile.create_dataset('NAME', data=np.asarray(NAME, dtype=dt), dtype=dt)
            datafile.create_dataset('SEQ', data=np.asarray(SEQ, dtype=dt), dtype=dt)
    else:
        datafile_path = f'{output_dir}datafile.txt'
        with open(datafile_path, 'w') as datafile: # temp sequence file
            for name, seq in zip(NAME, SEQ):
                datafile.write(f'{name}\n{seq}\n')
    
    return datafile_path, NAME, SEQ

    # check_and_count_motifs(gene_seq, labels, gene.strand) # maybe adapt to count motifs that were found in the predicted file...

def convert_sequences(datafile_path, output_dir, SEQ=None):
    '''
    Script to convert datafile into a one-hot encoded dataset ready to input to model. 
    If HDF5 file used, data is chunked for loading. 

    Parameters:
    - datafile_path: path to the datafile
    - output_dir: output directory path
    - SEQ: list of sequences 

    Returns:
    - Path to the dataset.
    '''

    # determine whether to convert an h5 or txt file
    file_ext = os.path.splitext(datafile_path)[1]
    assert file_ext in ['.h5', '.txt']
    use_h5 = file_ext == '.h5'

    # read the given input file if both datastreams were not provided
    if SEQ == None:
        print(f"\tReading {datafile_path} ... ")
        if use_h5:
            with h5py.File(datafile_path, 'r') as in_h5f:
                SEQ = in_h5f['SEQ'][:]
        else:
            SEQ = []
            with open(datafile_path, 'r') as in_file:
                lines = in_file.readlines()
                for i, line in enumerate(lines):
                    if i % 2 == 1: 
                        SEQ.append(line)
    else:
        print('\tNAME and SEQ data provided, skipping reading ...')

    num_seqs = len(SEQ)
    print("num_seqs: ", num_seqs)

    # write to h5 file by chunking and one-hot encoding inputs
    if use_h5:
        dataset_path = f'{output_dir}/dataset.h5'

        print(f"\tWriting {dataset_path} ... ")
        with h5py.File(dataset_path, 'w') as out_h5f:
               
            # Create dataset
            for i in range(num_seqs // CHUNK_SIZE):

                # Each dataset has CHUNK_SIZE genes
                if (i+1) == num_seqs // CHUNK_SIZE: # if last chunk, will add on all leftovers
                    NEW_CHUNK_SIZE = CHUNK_SIZE + num_seqs % CHUNK_SIZE
                else:
                    NEW_CHUNK_SIZE = CHUNK_SIZE

                X_batch = []
                for j in range(NEW_CHUNK_SIZE):
                    idx = i * CHUNK_SIZE + j

                    seq_decode = SEQ[idx]
                    X = create_datapoints(seq_decode)   
                    X_batch.extend(X)

                # Convert batches to arrays and save as HDF5
                X_batch = np.asarray(X_batch).astype('int8')
                # print("X_batch.shape: ", X_batch.shape)
                
                out_h5f.create_dataset('X' + str(i), data=X_batch)
    
    # convert to tensor and write directly to a binary PyTorch file for quick loading
    else:
        dataset_path = f'{output_dir}/dataset.pt'

        print(f"\tWriting {dataset_path} ... ")
        X_all = []
        for idx in range(num_seqs):
            seq_decode = SEQ[idx].decode('ascii')
            X = create_datapoints(seq_decode)
            X_all.extend(X)

        # convert batches to a tensor
        X_tensor = torch.tensor(X_all, dtype=torch.int8)

        # save as a binary file
        torch.save(X_tensor, dataset_path)     
    
    return dataset_path


def create_datapoints(input_string):
    """
    Parameters:
    - input_string (str): The nucleotide sequence.

    Returns:
    - X (np.ndarray): The one-hot encoded input nucleotide sequence.
    """
    global CL_max 

    # NOTE: No need to reverse complement the sequence, as sequence is already reverse complemented from previous step
    
    # Replace all non-ACTG to N
    allowed_chars = {'A', 'C', 'G', 'T'} # NOTE: this will turn all lowercase actg into N! (will not happen in here as seq already uppered)
    seq = ''.join(char if char in allowed_chars else 'N' for char in input_string) 
    
    # Convert to vector array
    seq = 'N' * (CL_max // 2) + seq + 'N' * (CL_max // 2)
    seq = seq.replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('N', '0')

    # One-hot-encode the inputs
    X0 = np.asarray(list(map(int, list(seq))))
    Xd = reformat_data(X0)
    X = one_hot_encode(Xd)

    return X

def reformat_data(X0):
    """
    Parameters:
    - X0 (numpy.ndarray): Original sequence data as an array of integer encodings.

    Returns:
    - numpy.ndarray: Reformatted sequence data.
    """
    global CL_max 

    # Calculate the number of data points needed
    num_points = ceil_div(len(X0), SL)
    # Initialize arrays to hold the reformatted data
    Xd = np.zeros((num_points, SL + CL_max))
    # Pad the sequence and labels to ensure divisibility
    X0 = np.pad(X0, (0, SL), 'constant', constant_values=0)
    
    # print(X0.shape, num_points, Xd.shape)    

    # Fill the initialized arrays with data in blocks
    for i in range(num_points):
        Xd[i] = X0[SL * i : SL * (i + 1) + CL_max]

    return Xd    

def one_hot_encode(Xd):
    """
    Perform one-hot encoding on both the input sequence data (Xd) and the output label data (Yd) using
    predefined mappings (IN_MAP for inputs and OUT_MAP for outputs).

    Parameters:
    - Xd (numpy.ndarray): An array of integers representing the input sequence data where each nucleotide
        is encoded as an integer (1 for 'A', 2 for 'C', 3 for 'G', 4 for 'T', and 0 for padding).

    Returns:
    - numpy.ndarray: the one-hot encoded input sequence data.
    """

    # One-hot encoding of the inputs: 
    # 1: A;  2: C;  3: G;  4: T;  0: padding
    IN_MAP = np.asarray([[0, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    return IN_MAP[Xd.astype('int8')]

####################
##   PREDICTION   ##
####################

# only used when file in hdf5
def load_shard(h5f, batch_size, shard_idx):
    '''
    Loads a selected shard from HDF5 file.

    Parameters: 
    - h5f: an OPEN dataset file in read mode
    '''

    X = h5f[f'X{shard_idx}'][:].transpose(0, 2, 1)
    X = torch.tensor(X, dtype=torch.float32)
    ds = TensorDataset(X)

    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

# custom for this implementation (removed dependency on Y)
def clip_datapoints(X, CL, N_GPUS):
    """
    This function is necessary to make sure of the following:
    (i) Each time model_m.fit is called, the number of datapoints is a
    multiple of N_GPUS. Failure to ensure this often results in crashes.
    (ii) If the required context length is less than CL_max, then
    appropriate clipping is done below.
    Additionally, Y is also converted to a list (the .h5 files store 
    them as an array).
    """
    global CL_max 
    
    print("\n\tX.shape: ", X.shape)
    print("\tCL: ", CL)
    print("\tN_GPUS: ", N_GPUS)

    rem = X.shape[0] % N_GPUS
    clip = (CL_max-CL)//2

    print("\trem: ", rem)
    print("\tclip: ", clip)

    if rem != 0 and clip != 0:
        return X[:-rem, :, clip:-clip]
    elif rem == 0 and clip != 0:
        return X[:, :, clip:-clip]
    elif rem != 0 and clip == 0:
        return X[:-rem]
    else:
        return X


# def model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode, criterion):
#     """
#     Evaluates the model's performance on a batch of data and logs the metrics.
#     Calculates various metrics, such as top-kL accuracy and AUPRC, for a given set of predictions and true labels.
#     The results are written to specified log files and can also be logged to Weights & Biases if enabled.

#     Parameters:
#     - batch_ylabel (list of torch.Tensor): A list of tensors containing the true labels for each batch.
#     - batch_ypred (list of torch.Tensor): A list of tensors containing the predicted labels for each batch.
#     - metric_files (dict): A dictionary containing paths to files where metrics should be logged.
#     - run_mode (str): The current phase of model usage ('train', 'validation', 'test') indicating where to log the metrics.
#     - criterion (str): The loss function that was used during training or evaluation, for appropriate metric calculation.
#     """

#     # batch_ylabel = torch.cat(batch_ylabel, dim=0)
#     batch_ypred = torch.cat(batch_ypred, dim=0)
#     is_expr = (batch_ylabel.sum(axis=(1,2)) >= 1).cpu().numpy()
#     if np.any(is_expr):
#         ############################
#         # Topk SpliceAI assessment approach
#         ############################
#         subset_size = 1000
#         indices = np.arange(batch_ylabel[is_expr].shape[0])
#         subset_indices = np.random.choice(indices, size=min(subset_size, len(indices)), replace=False)
#         # Y_true_1 = batch_ylabel[is_expr][subset_indices, 1, :].flatten().cpu().detach().numpy()
#         # Y_true_2 = batch_ylabel[is_expr][subset_indices, 2, :].flatten().cpu().detach().numpy()
#         Y_pred_1 = batch_ypred[is_expr][subset_indices, 1, :].flatten().cpu().detach().numpy()
#         Y_pred_2 = batch_ypred[is_expr][subset_indices, 2, :].flatten().cpu().detach().numpy()
#         acceptor_topkl_accuracy, acceptor_auprc = print_topl_statistics(np.asarray(Y_true_1),
#                             np.asarray(Y_pred_1), metric_files["topk_acceptor"], type='acceptor', print_top_k=True)
#         donor_topkl_accuracy, donor_auprc = print_topl_statistics(np.asarray(Y_true_2),
#                             np.asarray(Y_pred_2), metric_files["topk_donor"], type='donor', print_top_k=True)
#         # if criterion == "cross_entropy_loss":
#         #     loss = categorical_crossentropy_2d(batch_ylabel, batch_ypred)
#         # elif criterion == "focal_loss":
#         #     loss = focal_loss(batch_ylabel, batch_ypred)
#         for k, v in metric_files.items():
#             with open(v, 'a') as f:
#                 if k == "loss_batch":
#                     f.write(f"{loss.item()}\n")
#                 elif k == "topk_acceptor":
#                     f.write(f"{acceptor_topkl_accuracy}\n")
#                 elif k == "topk_donor":
#                     f.write(f"{donor_topkl_accuracy}\n")
#                 elif k == "auprc_acceptor":
#                     f.write(f"{acceptor_auprc}\n")
#                 elif k == "auprc_donor":
#                     f.write(f"{donor_auprc}\n")
#         wandb.log({
#             f'{run_mode}/loss_batch': loss.item(),
#             f'{run_mode}/topk_acceptor': acceptor_topkl_accuracy,
#             f'{run_mode}/topk_donor': donor_topkl_accuracy,
#             f'{run_mode}/auprc_acceptor': acceptor_auprc,
#             f'{run_mode}/auprc_donor': donor_auprc,
#         })
#         print("***************************************\n\n")
#     # batch_ylabel = []
#     batch_ypred = []


def get_prediction(model, dataset_path, criterion, device, params, metric_files, output_dir):
    """
    Parameters:
    - model (torch.nn.Module): The SpliceAI model to be evaluated.
    - dataset_path (path): Path to the selected dataset.
    - criterion (str): The loss function used for validation/testing.
    - device (torch.device): The computational device (CUDA, MPS, CPU).
    - params (dict): Dictionary of parameters related to model and validation/testing.
    - metric_files (dict): Dictionary containing paths to log files for various metrics.
    - output_dir (str): Root of output directory for predict file.

    Returns:
    - Path to predictions
    """
    # define batch size
    batch_size = params["BATCH_SIZE"]

    # put model in evaluation mode
    model.eval()

    # determine which file to proceed with
    file_ext = os.path.splitext(dataset_path)[1]
    assert file_ext in ['.h5', '.pt']
    use_h5 = file_ext == '.h5'

    # define used constants 

    # np.random.seed(RANDOM_SEED)  # You can choose any number as a seed
    # running_loss = 0.0
    
    batch_ypred = [] # list of tensors containing the predictions from model
    # print_dict = {}
    # will it fully predict on last batch? 

    if use_h5:
        h5f = h5py.File(dataset_path, 'r')

        # iterate over shards in index 
        idxs = np.arange(len(h5f.keys()) // 2)

        for i, shard_idx in enumerate(idxs, 1):
            print(f"Shard {i}/{len(idxs)}")
            loader = load_shard(h5f, batch_size, shard_idx)
            pbar = tqdm(loader, leave=False, total=len(loader), desc=f'Shard {i}/{len(idxs)}')
            for batch in pbar:
                DNAs = batch[0].to(device)

                print("\n\tDNAs.shape: ", DNAs.shape)
                DNAs = clip_datapoints(DNAs, params["CL"], params["N_GPUS"])
                DNAs = DNAs.to(torch.float32).to(device)
                print("\n\tAfter clipping DNAs.shape: ", DNAs.shape)

                y_pred = model(DNAs)

                # if criterion == "cross_entropy_loss":
                #     loss = categorical_crossentropy_2d(labels, y_pred)
                # elif criterion == "focal_loss":
                #     loss = focal_loss(labels, y_pred)
                    
                # Logging loss for every update!!! IMPORTANT
                # with open(metric_files["loss_every_update"], 'a') as f:
                #     f.write(f"{loss.item()}\n")
                # wandb.log({
                #     f'{run_mode}/loss_every_update': loss.item(),
                # })
                # running_loss += loss.item()

                # print("loss: ", loss.item())

                # batch_ylabel.append(labels.detach().cpu())

                batch_ypred.append(y_pred.detach().cpu())

                # print_dict["loss"] = loss.item()

                # pbar.set_postfix(print_dict)
                pbar.update(1)
            
            pbar.close()
    else:
        # all data should be loaded
        X = torch.load(dataset_file)
        X = torch.tensor(X, dtype=torch.float32)
        ds = TensorDataset(X)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

        pbar = tqdm(loader, leave=False, total=len(loader))
        for batch in pbar:
            DNAs = batch[0].to(device)

            print("\n\tDNAs.shape: ", DNAs.shape)
            DNAs = clip_datapoints(DNAs, params["CL"], params["N_GPUS"])
            DNAs = DNAs.to(torch.float32).to(device)
            print("\n\tAfter clipping DNAs.shape: ", DNAs.shape)

            y_pred = model(DNAs)

            batch_ypred.append(y_pred.detach().cpu())

            pbar.update(1)
        
        pbar.close()
    

    # model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode, criterion)

    # write all predictions to a predict_file
    predict_path = f'{output_dir}predict.pt'
    batch_ypred = torch.cat(batch_ypred, dim=0)
    torch.save(batch_ypred, predict_path)  
    return predict_path

def generate_bed(predict_file, NAME):
    ''' 
    Generates the BEDgraph file pertaining to the predictions 
    '''

    # set threshold for low values
    # write a separate file for donor and acceptor 
    # compress the ranges for same prediction score
    # raise NotImplementedError('generate_bed not yet implemented.')
    pass

################
##   DRIVER   ##
################

def predict(args):
    '''
    Parameters:
    - args (argparse.args): 
        - model: Path to SpliceAI model
        - output_dir
        - flanking_size
        - input_sequence: FASTA File

    '''
    print("Running SpliceAI-toolkit with 'predict' mode")
    # inputs args.: model, output_dir, flanking_size, input sequence (fasta file), 
    # outputs: the log files, bed files with scores for all splice sites

    # one-hot encode input sequence -> to DataLoader (as tensor) -> model.eval() -> get predictions (donor and acceptor sites only) / calculate loss
    # iterate over FASTA -> chunk -> if input sequence >5k, put in hdf5 format
    # generate BED file with scores for all splice sites -> visualize in IGV

    
    # PART 1: Extracting input sequence
    print("--- Step 1: Extracting input sequence ... ---")
    start_time = time.time()
    
    # get all input args
    output_dir = args.output_dir
    sequence_length = SL
    flanking_size = int(args.flanking_size)
    model_path = args.model
    input_sequence = args.input_sequence

    global CL_max 
    CL_max = flanking_size

    assert int(flanking_size) in [80, 400, 2000, 10000]

    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_base, log_output_base = initialize_paths(output_dir, flanking_size, sequence_length)
    print("* output_base: ", output_base, file=sys.stderr)
    print("* log_output_base: ", log_output_base, file=sys.stderr)
    print("Model path: ", model_path, file=sys.stderr)
    print("Flanking sequence size: ", flanking_size, file=sys.stderr)
    print("Sequence length: ", sequence_length, file=sys.stderr)

    # collect sequences into file
    datafile_path, NAME, SEQ = get_sequences(input_sequence, output_base)
    
    # print_motif_counts()

    print("--- %s seconds ---" % (time.time() - start_time))


    ### PART 2: Getting one-hot encoding of inputs
    print("--- Step 2: Creating one-hot encoding ... ---")
    start_time = time.time()

    dataset_path = convert_sequences(datafile_path, output_base, SEQ)

    print("--- %s seconds ---" % (time.time() - start_time))


    ### PART 3: Loading model
    print("--- Step 3: Load model ... ---")
    start_time = time.time()

    # setup device
    device = setup_device()
    print("device: ", device, file=sys.stderr)

    # load model from current state
    model, params = load_model(device, flanking_size)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    print("model: ", model, file=sys.stderr)
    print("params: ", params, file=sys.stderr)

    ## log files
    # predict_metric_files = {
    #     'topk_donor': f'{log_output_base}/donor_topk.txt',
    #     'auprc_donor': f'{log_output_base}/donor_accuracy.txt',
    #     'topk_acceptor': f'{log_output_base}/acceptor_topk.txt',
    #     'auprc_acceptor': f'{log_output_base}/acceptor_accuracy.txt',
    #     'loss_batch': f'{log_output_base}/loss_batch.txt',
    #     'loss_every_update': f'{log_output_base}/loss_every_update.txt' #only rly important one!
    # } 

    print("--- %s seconds ---" % (time.time() - start_time))


    ### PART 4: Get predictions
    print("--- Step 4: Get predictions ... ---")
    start_time = time.time()

    # for testing
    predict_metric_files = None
    criterion = None

    predict_file = get_prediction(model, dataset_path, criterion, device, 
                                    params, predict_metric_files, output_base)


    print("--- %s seconds ---" % (time.time() - start_time))


    ### PART 5: Generate BEDgraph report
    print("--- Step 5: Generating BEDgraph report ... ---")
    start_time = time.time()

    generate_bed(predict_file, NAME)

    print("--- %s seconds ---" % (time.time() - start_time))

    # other function: give BED coordinate file (optional arg) -> extract just those sequences from FASTA for prediction
    #                   give GFF file + and what you want to extract -> go to gene-level 
    #                   --coordinates (BED/GFF format) -> detect which extension
    # genes[id][start:end] 



### TODO: tests
# predict on /models/spliceai-mane
# input gff file, extract gene-level sequences, then predict score for each position in gene
# later -> bedgraph file

# if __name__ == '__main__':

    # CL_max=400
    # # Maximum nucleotide context length (CL_max/2 on either side of the 
    # # position of interest)
    # # CL_max should be an even number

    # SL=5000
    # # Sequence length of SpliceAIs (SL+CL will be the input length and
    # # SL will be the output length)
    # #############################
    # # Global variable definition
    # ############################## 
    # EPOCH_NUM = 10

    
    # parser = argparse.ArgumentParser(description='SpliceAI toolkit to retrain your own splice site predictor')
    # # Create a parent subparser to house the common subcommands.
    # subparsers = parser.add_subparsers(dest='command', required=True, help='Subcommands: create-data, train, predict, variant')
    # parser_predict = subparsers.add_parser('predict', help='Predict splice sites in a given sequence using the SpliceAI model')
    # parser_predict.add_argument('--model', '-m', default="SpliceAI", type=str)
    # parser_predict.add_argument('--output-dir', '-o', type=str, required=True, help='Output directory to save the data')
    # parser_predict.add_argument('--flanking-size', '-f', type=int, default=80, help='Sum of flanking sequence lengths on each side of input (i.e. 40+40)')
    # parser_predict.add_argument('--input-sequence', '-i', type=str, help="Path to FASTA file of the input sequence")
    # args = parser.parse_args()
    # print(args)
    # predict(args)         

    
