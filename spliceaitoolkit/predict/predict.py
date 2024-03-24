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
from spliceaitoolkit.predict.spliceai import *
from spliceaitoolkit.predict.utils import *
from spliceaitoolkit.constants import *
from Bio import SeqIO
import wandb

RANDOM_SEED = 42 # for replicability
HDF_THRESHOLD_LEN = 5000 # maximum size before reading sequence into an HDF file for storage

#####################
##      SETUP      ##
#####################

# initialize output directory for log files, predict.bed file
def initialize_paths(output_dir, flanking_size, sequence_length, model_arch):
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
def load_model(device, flanking_size, model_arch):
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
    params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE}

    print(model, file=sys.stderr)
    return model, params


#########################
##   DATA PROCESSING   ##
#########################
    
'''MODIFY TO JUST GET SEQUENCES, BUILD SEP FUNCTION THAT READS INTO TEMPFILE IF SIZE <=5K'''
def get_sequences(fasta_file, output_dir):
    """
    Extract sequences for each protein-coding gene, process them based on strand orientation,
    and save data in file depending on the sequence size (HDF file if sequence >HDF_THRESHOLD_LEN, 
    else temp file).

    Parameters:
    - fasta_file: Path to the FASTA file.
    - output_dir: Directory to save the output files.

    Returns:
    - Path to output file
    """

    # detect sequence length
    total_length = 0
    use_hdf = False
    for record in SeqIO.parse(fasta_file, "fasta"):
        total_length += len(record.seq)
        if total_length > HDF_THRESHOLD_LEN:
            use_hdf = True
            break

    # write sequence information to temp custom file
    if not use_hdf:
        pass
    
    # write sequence to compressed hdf format
    seq_dict = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    fw_stats = open(f"{output_dir}stats.txt", "w") # statistics file
    h5f = h5py.File(output_dir + f'datafile_{type}.h5', 'w') # hdf5 information file

    NAME = []      # Gene Name
    CHROM = []     # Chromosome
    STRAND = []    # Strand in which the gene lies (+ or -)
    TX_START = []  # Position where transcription starts
    TX_END = []    # Position where transcription ends
    SEQ = []       # Nucleotide sequence
    LABEL = []     # Label for each nucleotide in the sequence
    GENE_COUNTER = 0

    for record in SeqIO.parse(fasta_file, "fasta"):
        header = record.id
        print(header)
        # Assuming the header is formatted like: chromosome:start-end
        # Adjust this parsing to fit your specific header format
        try:
            chromosome, positions = header.split(":")
            start, end = positions.split("-")
        except ValueError:
            print(f"Skipping {header}: cannot parse into chrom, start, and end.")

    # no gff file? can't just do this
    for gene in db.features_of_type('gene'):
        if gene.attributes["gene_biotype"][0] == "protein_coding" and gene.seqid in chrom_dict:
            chrom_dict[gene.seqid] += 1 
            gene_id = gene.id 
            gene_seq = seq_dict[gene.seqid].seq[gene.start-1:gene.end].upper()  # Extract gene sequence
            labels = [0] * len(gene_seq)  # Initialize all labels to 0
            transcripts = list(db.children(gene, featuretype='mRNA', order_by='start'))
            if len(transcripts) == 0:
                continue
            elif len(transcripts) > 1:
                print(f"Gene {gene_id} has multiple transcripts: {len(transcripts)}")
            ############################################
            # Selecting which mode to process the data
            ############################################
            transcripts_ls = []
            if parse_type == 'maximum':
                max_trans = transcripts[0]
                max_len = max_trans.end - max_trans.start + 1
                for transcript in transcripts:
                    if transcript.end - transcript.start + 1 > max_len:
                        max_trans = transcript
                        max_len = transcript.end - transcript.start + 1
                transcripts_ls = [max_trans]
            elif parse_type == 'all_isoforms': # leave in?
                transcripts_ls = transcripts
            # Process transcripts
            for transcript in transcripts_ls:
                exons = list(db.children(transcript, featuretype='exon', order_by='start'))
                if len(exons) > 1:
                    GENE_COUNTER += 1
                    for i in range(len(exons) - 1):
                        # Donor site is one base after the end of the current exon
                        first_site = exons[i].end - gene.start  # Adjusted for python indexing
                        # Acceptor site is at the start of the next exon
                        second_site = exons[i + 1].start - gene.start  # Adjusted for python indexing
                        if gene.strand == '+':
                            labels[first_site] = 2  # Mark donor site
                            labels[second_site] = 1  # Mark acceptor site
                        elif gene.strand == '-':
                            d_idx = len(labels) - second_site-1
                            a_idx = len(labels) - first_site-1
                            labels[d_idx] = 2   # Mark donor site
                            labels[a_idx] = 1  # Mark acceptor site
                            seq = gene_seq.reverse_complement()
                            print("D: ", seq[d_idx-3:  d_idx+4])
                            print("A: ", seq[a_idx-6: a_idx+3])
            if gene.strand == '-':
                gene_seq = gene_seq.reverse_complement() # reverse complement the sequence
            gene_seq = str(gene_seq.upper())
            labels_str = ''.join(str(num) for num in labels)
            NAME.append(gene_id)
            CHROM.append(gene.seqid)
            STRAND.append(gene.strand)
            TX_START.append(str(gene.start))
            TX_END.append(str(gene.end))
            SEQ.append(gene_seq)
            LABEL.append(labels_str)
            fw_stats.write(f"{gene.seqid}\t{gene.start}\t{gene.end}\t{gene.id}\t{1}\t{gene.strand}\n")
            check_and_count_motifs(gene_seq, labels, gene.strand)
    fw_stats.close()
    dt = h5py.string_dtype(encoding='utf-8')
    h5f.create_dataset('NAME', data=np.asarray(NAME, dtype=dt) , dtype=dt)
    h5f.create_dataset('CHROM', data=np.asarray(CHROM, dtype=dt) , dtype=dt)
    h5f.create_dataset('STRAND', data=np.asarray(STRAND, dtype=dt) , dtype=dt)
    h5f.create_dataset('TX_START', data=np.asarray(TX_START, dtype=dt) , dtype=dt)
    h5f.create_dataset('TX_END', data=np.asarray(TX_END, dtype=dt) , dtype=dt)
    h5f.create_dataset('SEQ', data=np.asarray(SEQ, dtype=dt) , dtype=dt)
    h5f.create_dataset('LABEL', data=np.asarray(LABEL, dtype=dt) , dtype=dt)
    h5f.close()   


def create_datapoints(input_string):
    """
    Parameters:
    - input_string (str): The nucleotide sequence.

    Returns:
    - X (np.ndarray): The one-hot encoded input nucleotide sequence.
    """

    # NOTE: No need to reverse complement the sequence, as sequence is already reverse complemented from previous step
    
    # Replace all non-ACTG to N
    allowed_chars = {'A', 'C', 'G', 'T'} # NOTE: this will turn all lowercase actg into N!
    seq = ''.join(char if char in allowed_chars else 'N' for char in input_string) 
    
    # Convert to vector array
    seq = 'N' * (CL_max // 2) + seq + 'N' * (CL_max // 2)
    seq = seq.replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('N', '0')

    # One-hot-encode the inputs
    X0 = np.asarray(list(map(int, list(seq))))
    Xd = reformat_data(X0)
    X = one_hot_encode(Xd)

    return X

'''REMOVED ALL DEPENDENCY ON LABEL FOR PREDICT.PY'''
def reformat_data(X0):
    """
    Parameters:
    - X0 (numpy.ndarray): Original sequence data as an array of integer encodings.

    Returns:
    - numpy.ndarray: Reformatted sequence data.
    """
    # Calculate the number of data points needed
    num_points = ceil_div(len(Y0[0]), SL)
    # Initialize arrays to hold the reformatted data
    Xd = np.zeros((num_points, SL + CL_max))
    # Pad the sequence and labels to ensure divisibility
    X0 = np.pad(X0, (0, SL), 'constant', constant_values=0)
    # Fill the initialized arrays with data in blocks
    for i in range(num_points):
        Xd[i] = X0[SL * i : SL * (i + 1) + CL_max]

    return Xd    

# One-hot encoding of the inputs: 
# 1: A;  2: C;  3: G;  4: T;  0: padding
IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

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
    return IN_MAP[Xd.astype('int8')]


'''if input sequence >5k, put into hdf5 format'''
def load_data_from_shard(h5f, shard_idx, device, batch_size, params, shuffle=False):
    X = h5f[f'X{shard_idx}'][:].transpose(0, 2, 1)
    Y = h5f[f'Y{shard_idx}'][0, ...].transpose(0, 2, 1)
    # print("\n\tX.shape: ", X.shape)
    # print("\tY.shape: ", Y.shape)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    ds = TensorDataset(X, Y)

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, pin_memory=True)

####################
##   PREDICTION   ##
####################

def model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode, criterion):
    """
    Evaluates the model's performance on a batch of data and logs the metrics.
    Calculates various metrics, such as top-kL accuracy and AUPRC, for a given set of predictions and true labels.
    The results are written to specified log files and can also be logged to Weights & Biases if enabled.

    Parameters:
    - batch_ylabel (list of torch.Tensor): A list of tensors containing the true labels for each batch.
    - batch_ypred (list of torch.Tensor): A list of tensors containing the predicted labels for each batch.
    - metric_files (dict): A dictionary containing paths to files where metrics should be logged.
    - run_mode (str): The current phase of model usage ('train', 'validation', 'test') indicating where to log the metrics.
    - criterion (str): The loss function that was used during training or evaluation, for appropriate metric calculation.
    """

    batch_ylabel = torch.cat(batch_ylabel, dim=0)
    batch_ypred = torch.cat(batch_ypred, dim=0)
    is_expr = (batch_ylabel.sum(axis=(1,2)) >= 1).cpu().numpy()
    if np.any(is_expr):
        ############################
        # Topk SpliceAI assessment approach
        ############################
        subset_size = 1000
        indices = np.arange(batch_ylabel[is_expr].shape[0])
        subset_indices = np.random.choice(indices, size=min(subset_size, len(indices)), replace=False)
        Y_true_1 = batch_ylabel[is_expr][subset_indices, 1, :].flatten().cpu().detach().numpy()
        Y_true_2 = batch_ylabel[is_expr][subset_indices, 2, :].flatten().cpu().detach().numpy()
        Y_pred_1 = batch_ypred[is_expr][subset_indices, 1, :].flatten().cpu().detach().numpy()
        Y_pred_2 = batch_ypred[is_expr][subset_indices, 2, :].flatten().cpu().detach().numpy()
        acceptor_topkl_accuracy, acceptor_auprc = print_topl_statistics(np.asarray(Y_true_1),
                            np.asarray(Y_pred_1), metric_files["topk_acceptor"], type='acceptor', print_top_k=True)
        donor_topkl_accuracy, donor_auprc = print_topl_statistics(np.asarray(Y_true_2),
                            np.asarray(Y_pred_2), metric_files["topk_donor"], type='donor', print_top_k=True)
        if criterion == "cross_entropy_loss":
            loss = categorical_crossentropy_2d(batch_ylabel, batch_ypred)
        elif criterion == "focal_loss":
            loss = focal_loss(batch_ylabel, batch_ypred)
        for k, v in metric_files.items():
            with open(v, 'a') as f:
                if k == "loss_batch":
                    f.write(f"{loss.item()}\n")
                elif k == "topk_acceptor":
                    f.write(f"{acceptor_topkl_accuracy}\n")
                elif k == "topk_donor":
                    f.write(f"{donor_topkl_accuracy}\n")
                elif k == "auprc_acceptor":
                    f.write(f"{acceptor_auprc}\n")
                elif k == "auprc_donor":
                    f.write(f"{donor_auprc}\n")
        wandb.log({
            f'{run_mode}/loss_batch': loss.item(),
            f'{run_mode}/topk_acceptor': acceptor_topkl_accuracy,
            f'{run_mode}/topk_donor': donor_topkl_accuracy,
            f'{run_mode}/auprc_acceptor': acceptor_auprc,
            f'{run_mode}/auprc_donor': donor_auprc,
        })
        print("***************************************\n\n")
    batch_ylabel = []
    batch_ypred = []


# definitely need
def valid_epoch(model, h5f, idxs, batch_size, criterion, device, params, metric_files, run_mode, sample_freq):
    """
    Validates the SpliceAI model on a given dataset.
    (Similar to train_epoch, but without performing backpropagation or updating model parameters)

    Parameters:
    - model (torch.nn.Module): The SpliceAI model to be evaluated.
    - h5f (h5py.File): HDF5 file object containing the validation or test data.
    - idxs (np.array): Array of indices for the batches to be used in validation/testing.
    - batch_size (int): Size of each batch.
    - criterion (str): The loss function used for validation/testing.
    - device (torch.device): The computational device (CUDA, MPS, CPU).
    - params (dict): Dictionary of parameters related to model and validation/testing.
    - metric_files (dict): Dictionary containing paths to log files for various metrics.
    - run_mode (str): Indicates the phase (e.g., "validation", "test").
    - sample_freq (int): Frequency of sampling for evaluation and logging.
    """

    print(f"\033[1m{run_mode.capitalize()}ing model...\033[0m")
    # put model in evaluation mode
    model.eval()

    running_loss = 0.0
    np.random.seed(RANDOM_SEED)  # You can choose any number as a seed

    shuffled_idxs = np.random.choice(idxs, size=len(idxs), replace=False)    
    print("shuffled_idxs: ", shuffled_idxs)
    batch_ylabel = []
    batch_ypred = []
    print_dict = {}
    batch_idx = 0
    for i, shard_idx in enumerate(shuffled_idxs, 1):
        print(f"Shard {i}/{len(shuffled_idxs)}")
        loader = load_data_from_shard(h5f, shard_idx, device, batch_size, params, shuffle=False)
        pbar = tqdm(loader, leave=False, total=len(loader), desc=f'Shard {i}/{len(shuffled_idxs)}')
        for batch in pbar:
            DNAs, labels = batch[0].to(device), batch[1].to(device)
            # print("\n\tDNAs.shape: ", DNAs.shape)
            # print("\tlabels.shape: ", labels.shape)
            DNAs, labels = clip_datapoints(DNAs, labels, params["CL"], 2)
            DNAs, labels = DNAs.to(torch.float32).to(device), labels.to(torch.float32).to(device)
            # print("\n\tAfter clipping DNAs.shape: ", DNAs.shape)
            # print("\tAfter clipping labels.shape: ", labels.shape)
            yp = model(DNAs)
            if criterion == "cross_entropy_loss":
                loss = categorical_crossentropy_2d(labels, yp)
            elif criterion == "focal_loss":
                loss = focal_loss(labels, yp)
                
            # Logging loss for every update!!! IMPORTANT
            with open(metric_files["loss_every_update"], 'a') as f:
                f.write(f"{loss.item()}\n")
            wandb.log({
                f'{run_mode}/loss_every_update': loss.item(),
            })
            running_loss += loss.item()

            # print("loss: ", loss.item())
            batch_ylabel.append(labels.detach().cpu())
            batch_ypred.append(yp.detach().cpu())
            print_dict["loss"] = loss.item()
            pbar.set_postfix(print_dict)
            pbar.update(1)
            batch_idx += 1
            # if batch_idx % sample_freq == 0:
            #     model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode)
        pbar.close()
    model_evaluation(batch_ylabel, batch_ypred, metric_files, run_mode, criterion)


################
##   DRIVER   ##
################

def predict(args):
    '''
    Parameters:
    - args (argparse.args): 
        - model: SpliceAI (default)
        - output_dir
        - flanking_size
        - input_sequence

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
    model_arch = args.model
    input_sequence = args.input_sequence

    assert int(flanking_size) in [80, 400, 2000, 10000]
   
    # setup
    device = setup_device()
    print("device: ", device, file=sys.stderr)

    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_base, log_output_base = initialize_paths(output_dir, flanking_size, sequence_length, model_arch)
    print("* Project name: ", args.project_name, file=sys.stderr)
    print("* log_output_base: ", log_output_base, file=sys.stderr)

    print("Model architecture: ", model_arch, file=sys.stderr)
    print("Flanking sequence size: ", args.flanking_size, file=sys.stderr)

    # collect sequences
    _,_,_ = get_sequences(input_sequence, output_dir)
    
    print_motif_counts()

    print("--- %s seconds ---" % (time.time() - start_time))

    ### PART 2: converting to tensor foramt

    print("--- Step 2: Creating dataset.h5 ... ---")
    start_time = time.time()
    for data_type in ['train', 'test']:
        print(("\tProcessing %s ..." % data_type))
        input_file = f"{args.output_dir}/datafile_{data_type}.h5"
        output_file = f"{args.output_dir}/dataset_{data_type}.h5"

        print(f"\tReading {input_file} ... ")
        with h5py.File(input_file, 'r') as h5f:
            SEQ = h5f['SEQ'][:]
            LABEL = h5f['LABEL'][:]
            STRAND = h5f['STRAND'][:]
            TX_START = h5f['TX_START'][:]
            TX_END = h5f['TX_END'][:]
            SEQ = h5f['SEQ'][:]
            LABEL = h5f['LABEL'][:]

        print(f"\tWriting {output_file} ... ")
        with h5py.File(output_file, 'w') as h5f2:
            seq_num = len(SEQ)
            CHUNK_SIZE = 100

            print("seq_num: ", seq_num)
            print("STRAND.shape[0]: ", STRAND.shape[0])
            print("TX_START.shape[0]: ", TX_START.shape[0])
            print("TX_END.shape[0]: ", TX_END.shape[0])
            print("LABEL.shape[0]: ", LABEL.shape[0])

            # Create dataset
            for i in range(seq_num // CHUNK_SIZE):

                # Each dataset has CHUNK_SIZE genes
                if (i+1) == seq_num // CHUNK_SIZE:
                    NEW_CHUNK_SIZE = CHUNK_SIZE + seq_num%CHUNK_SIZE
                else:
                    NEW_CHUNK_SIZE = CHUNK_SIZE

                X_batch, Y_batch = [], [[] for _ in range(1)]

                for j in range(NEW_CHUNK_SIZE):
                    idx = i * CHUNK_SIZE + j

                    seq_decode = SEQ[idx].decode('ascii')
                    strand_decode = STRAND[idx].decode('ascii')
                    tx_start_decode = TX_START[idx].decode('ascii')
                    tx_end_decode = TX_END[idx].decode('ascii')
                    label_decode = LABEL[idx].decode('ascii')

                    X = create_datapoints(seq_decode, strand_decode, label_decode)   

                    X_batch.extend(X)

                # Convert batches to arrays and save as HDF5
                X_batch = np.asarray(X_batch).astype('int8')
                print("X_batch.shape: ", X_batch.shape)
             
                h5f2.create_dataset('X' + str(i), data=X_batch)

    print("--- %s seconds ---" % (time.time() - start_time))

    # batch_num = len(train_h5f.keys()) // 2
    # print("Batch_num: ", batch_num, file=sys.stderr)
    # np.random.seed(RANDOM_SEED)  # You can choose any number as a seed
    # idxs = np.random.permutation(batch_num)
    # train_idxs = idxs[:int(0.9 * batch_num)]
    # val_idxs = idxs[int(0.9 * batch_num):]
    # test_idxs = np.arange(len(test_h5f.keys()) // 2)
    # print("train_idxs: ", train_idxs, file=sys.stderr)
    # print("val_idxs: ", val_idxs, file=sys.stderr)
    # print("test_idxs: ", test_idxs, file=sys.stderr)

    model, params = load_model(device, flanking_size, model_arch)

    ## log files
    predict_metric_files = {
        'topk_donor': f'{log_output_base}/donor_topk.txt',
        'auprc_donor': f'{log_output_base}/donor_accuracy.txt',
        'topk_acceptor': f'{log_output_base}/acceptor_topk.txt',
        'auprc_acceptor': f'{log_output_base}/acceptor_accuracy.txt',
        'loss_batch': f'{log_output_base}/loss_batch.txt',
        'loss_every_update': f'{log_output_base}/loss_every_update.txt' #only rly important one!
    } 

    SAMPLE_FREQ = 1000
    for epoch in range(EPOCH_NUM):
        print("\n--------------------------------------------------------------")
        print(f">> Epoch {epoch + 1}")
        start_time = time.time()
        valid_epoch(model, train_h5f, val_idxs, params["BATCH_SIZE"], args.loss, device, params, predict_metric_files, run_mode="validation", sample_freq=SAMPLE_FREQ)
        torch.save(model.state_dict(), f"{model_output_base}/model_{epoch}.pt")
        print("--- %s seconds ---" % (time.time() - start_time))
        print("--------------------------------------------------------------")

    #??? 
    valid_epoch(model)
