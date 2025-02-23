import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
from pyfaidx import Fasta
import logging
import platform
import sys, os, glob
import torch
from openspliceai.train_base.spliceai import SpliceAI
from openspliceai.constants import *
import logomaker
import matplotlib.pyplot as plt
import math

import itertools
from keras import backend as K
from tensorflow import keras

##############################################
## LOADING PYTORCH AND KERAS MODELS
##############################################

def setup_device():
        """Select computation device based on availability."""
        device_str = "cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" else "cpu"
        return torch.device(device_str)

def load_pytorch_models(model_path, CL):
    """
    Loads a SpliceAI PyTorch model from given state, inferring device.
    
    Params:
    - model_path (str): Path to the model state dict, or a directory of models
    - CL (int): Context length parameter for model conversion.
    
    Returns:
    - loaded_models (list): SpliceAI model(s) loaded with given state.
    """
    
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

        print(f"\t[INFO] Context nucleotides {CL}")
        print(f"\t[INFO] Sequence length (output): {SL}")
        
        model = SpliceAI(L, W, AR).to(device)
        params = {'L': L, 'W': W, 'AR': AR, 'CL': CL, 'SL': SL, 'BATCH_SIZE': BATCH_SIZE, 'N_GPUS': N_GPUS}

        return model, params
    
    # Setup device
    device = setup_device()
    
    # Load all model state dicts given the supplied model path
    if os.path.isdir(model_path):
        model_files = glob.glob(os.path.join(model_path, '*.pth')) # gets all PyTorch models from supplied directory
        if not model_files:
            logging.error(f"No PyTorch model files found in directory: {model_path}")
            exit()
            
        models = []
        for model_file in model_files:
            try:
                model = torch.load(model_file, map_location=device)
                models.append(model)
            except Exception as e:
                logging.error(f"Error loading PyTorch model from file {model_file}: {e}. Skipping...")
                
        if not models:
            logging.error(f"No valid PyTorch models found in directory: {model_path}")
            exit()
    
    elif os.path.isfile(model_path):
        try:
            models = [torch.load(model_path, map_location=device)]
        except Exception as e:
            logging.error(f"Error loading PyTorch model from file {model_path}: {e}.")
            exit()
        
    else:
        logging.error(f"Invalid path: {model_path}")
        exit()
    
    # Load state of model to device
    # NOTE: supplied model paths should be state dicts, not model files  
    loaded_models = []
    
    for state_dict in models:
        try: 
            model, params = load_model(device, CL) # loads new SpliceAI model with correct hyperparams
            model.load_state_dict(state_dict)      # loads state dict
            model = model.to(device)               # puts model on device
            model.eval()                           # puts model in evaluation mode
            loaded_models.append(model)            # appends model to list of loaded models  
        except Exception as e:
            logging.error(f"Error processing model for device: {e}. Skipping...")
            
    if not loaded_models:
        logging.error("No models were successfully loaded to the device.")
        exit()
        
    return loaded_models, device

def load_keras_models(model_path):
    """
    Loads Keras models from given path.
    
    Params:
    - model_path (str): Path to the model file or directory of models.
    
    Returns:
    - models (list): List of loaded Keras models.
    """
    if os.path.isdir(model_path): # directory supplied
        model_files = glob.glob(os.path.join(model_path, '*.h5')) # get all Keras models from a directory
        if not model_files:
            logging.error(f"No Keras model files found in directory: {model_path}")
            exit()
            
        models = []
        for model_file in model_files:
            try:
                model = keras.models.load_model(model_file)
                models.append(model)
            except Exception as e:
                logging.error(f"Error loading Keras model from file {model_file}: {e}. Skipping...")
                
        if not models:
            logging.error(f"No valid PyTorch models found in directory: {model_path}")
            exit()
            
        return models, None
    
    elif os.path.isfile(model_path): # file supplied
        try:
            return keras.models.load_model(model_path)
        except Exception as e:
            logging.error(f"Error loading Keras model from file {model_path}: {e}")
            exit()
        
    else: # invalid path
        logging.error(f"Invalid path: {model_path}")
        exit()

# Updated load function for models (Keras and PyTorch models)
def load_models(model_path, model_type, CL):
    """
    Loads models based on model type (Keras/PyTorch). Can take SpliceAI path as argument for keras.
    Returns: Model paths, Device
    """
    if model_type == "pytorch":
        return load_pytorch_models(model_path, CL)
    elif model_type == "keras":
        if model_path == 'SpliceAI':
            paths = ('./models/spliceai/spliceai{}.h5'.format(x) for x in range(1, 6))
            return [keras.models.load_model(x) for x in paths], None
        elif model_path == None:
            paths = (f'models/SpliceAI/SpliceNet{CL}_c{x}.h5' for x in range(1, 6))
            models = [load_keras_models(x) for x in paths]
            return models, None
        return load_keras_models(model_path)
    else:
        logging.error("Invalid model type specified")
        exit()

##############################################
## PREDICT FUNCTIONS
##############################################

def one_hot_encode(seq):
    """
    One-hot encode a DNA sequence.
    
    Args:
        seq (str): DNA sequence to be encoded.
    
    Returns:
        np.ndarray: One-hot encoded representation of the sequence.
    """

    # Define a mapping matrix for nucleotide to one-hot encoding
    IN_MAP = np.asarray([[0, 0, 0, 0],  # N or any invalid character
                      [1, 0, 0, 0],  # A
                      [0, 1, 0, 0],  # C
                      [0, 0, 1, 0],  # G
                      [0, 0, 0, 1]]) # T

    # Replace nucleotides with corresponding indices
    seq = seq.upper().replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')
    X0 = np.asarray(list(map(int, list(seq)))).astype('int8')

    # Convert the sequence to one-hot encoded numpy array
    return IN_MAP[X0 % 5]

def predict_keras(models, flanking_size, seq, strand='+', padding=False):
    # Prepare the sequence with padding
    if padding:
        pad_size = [flanking_size // 2, flanking_size // 2]
        x = 'N' * pad_size[0] + seq + 'N' * pad_size[1]
    else:
        x = seq

    # One-hot encode the sequence
    x = one_hot_encode(x)[None, :]

    # Reverse the sequence if on the negative strand
    if strand == '-':
        x = x[:, ::-1, ::-1]

    # Predict the scores using the models
    y = np.mean([models[m].predict(x) for m in range(len(models))], axis=0)

    # Reverse the scores if on the negative strand
    if strand == '-':
        y = y[:, ::-1]

    # Extract donor and acceptor scores
    acceptor_scores = y[0, :, 1]
    donor_scores = y[0, :, 2]
    
    return acceptor_scores, donor_scores

def predict_pytorch(models, flanking_size, seq, strand='+', device='cuda', padding=False):
    # Prepare the sequence with padding
    if padding:
        pad_size = [flanking_size // 2, flanking_size // 2]
        x = 'N' * pad_size[0] + seq + 'N' * pad_size[1]
    else:
        x = seq

    # One-hot encode the sequence
    x = one_hot_encode(x)[None, :]

    # Transpose to match PyTorch input dimensions
    x = x.transpose(0, 2, 1)

    # Convert to PyTorch tensor
    x = torch.tensor(x, dtype=torch.float32).to(device)

    # Reverse the sequence if on the negative strand
    if strand == '-':
        x = torch.flip(x, dims=[1, 2])

    # Predict the scores using the models
    with torch.no_grad():
        y = torch.mean(torch.stack([models[m](x).detach().cpu() for m in range(len(models))]), axis=0)
        
    # Remove flanking sequence and permute shape
    y = y.permute(0, 2, 1)

    # Reverse the scores if on the negative strand
    if strand == '-':
        y = torch.flip(y, dims=[1])

    # Extract donor and acceptor scores
    y = y.numpy()
    
    acceptor_scores = y[0, :, 1]
    donor_scores = y[0, :, 2]
    
    return acceptor_scores, donor_scores

def predict(models, model_type, flanking_size, seq, strand='+', device='cuda', padding=False):
    if model_type == 'keras':
        return predict_keras(models, flanking_size, seq, strand, padding)
    elif model_type == 'pytorch':
        return predict_pytorch(models, flanking_size, seq, strand, device, padding)
    else:
        logging.error("Invalid model type")
        exit()
        
##############################################
## UTILS: ONE-HOT ENCODING, MUTATION, LOGOS
##############################################

# Function to mutate a sequence of bases to all other possible bases
def get_mutations(bases_to_mutate):
    mutations = list(itertools.product(['A', 'C', 'G', 'T'], repeat=len(bases_to_mutate)))
    mutations = [m for m in mutations if m != bases_to_mutate]
    return mutations

# Function to generate DNA logo
def generate_dna_logo(score_changes_df, output_file, start=None, end=None):
    """
    Generates a DNA logo from the score changes DataFrame.
    """
    # Slice the DataFrame to include only rows from start to end
    if start is not None and end is not None:
        data_df = score_changes_df.iloc[start:end]
    else:
        data_df = score_changes_df
    
    # Ensure valid data
    data_df = data_df.fillna(0)
    
    # Create the logo
    plt.figure(figsize=(12, 4))
    logo = logomaker.Logo(data_df)
    logo.ax.set_title(f'OpenSpliceAI Score Change by Base')
    logo.ax.set_ylabel('Score Change')
    logo.ax.set_xlabel('Position')
    plt.savefig(output_file)
    plt.close()
    
def plot_donor_acceptor_sites(acceptor_scores_ref, acceptor_scores_mut, donor_scores_ref, donor_scores_mut, output_dir):
    positions = range(len(acceptor_scores_ref))
    
    plt.figure(figsize=(12, 6))
    
    # Plot acceptor scores
    plt.subplot(2, 1, 1)
    plt.plot(positions, acceptor_scores_ref, label='Acceptor Ref', color='blue', alpha=0.5)
    plt.plot(positions, donor_scores_ref, label='Donor Ref', color='red', alpha=0.5)
    plt.title('Reference')
    plt.xlabel('Position')
    plt.ylabel('Score')
    plt.legend()
    
    # Plot donor scores
    plt.subplot(2, 1, 2)
    plt.plot(positions, acceptor_scores_mut, label='Acceptor Mut', color='blue', alpha=0.5)
    plt.plot(positions, donor_scores_mut, label='Donor Mut', color='red', alpha=0.5)
    plt.title('Mutated')
    plt.xlabel('Position')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'donor_acceptor_sites.png'))
    plt.show()

##############################################
## MUTAGENESIS EXPERIMENT
##############################################

def exp_1(fasta_file, models, model_type, flanking_size, output_dir, device, mutation_position, mutation_base):
    
    # Load fasta file
    fasta = Fasta(fasta_file)
    header = list(fasta.keys())[0]
    sequence = str(fasta[header])
    print(len(sequence))
    
    #### PADDING
    padding=False

    # Get the reference base sequence scores
    acceptor_scores_ref, donor_scores_ref = predict(models, model_type, flanking_size, sequence, device=device, padding=padding)

    # Convert scores to numpy arrays if they're tensors
    if not isinstance(acceptor_scores_ref, np.ndarray):
        acceptor_scores_ref = acceptor_scores_ref.cpu().numpy()
    if not isinstance(donor_scores_ref, np.ndarray):
        donor_scores_ref = donor_scores_ref.cpu().numpy()

    # Modify the sequence by changing the base at mutation_position to the mutation_base
    mutated_sequence = list(sequence)
    mutated_sequence[mutation_position] = mutation_base
    mutated_sequence = ''.join(mutated_sequence) 

    # Predict scores for the mutated sequence
    acceptor_scores_mut, donor_scores_mut = predict(models, model_type, flanking_size, mutated_sequence, device=device, padding=padding)

    # Convert mutated scores to numpy arrays if they're tensors
    if not isinstance(acceptor_scores_mut, np.ndarray):
        acceptor_scores_mut = acceptor_scores_mut.cpu().numpy()
    if not isinstance(donor_scores_mut, np.ndarray):
        donor_scores_mut = donor_scores_mut.cpu().numpy()

    # Save the donor and acceptor scores into a csv file
    scores_df = pd.DataFrame({
        'Position': range(len(acceptor_scores_ref)),
        'Acceptor_Ref': acceptor_scores_ref,
        'Donor_Ref': donor_scores_ref,
        'Acceptor_Mut': acceptor_scores_mut,
        'Donor_Mut': donor_scores_mut
    })
    scores_df.to_csv(os.path.join(output_dir, 'scores.csv'), index=False)
    
    # Compute score differences
    acceptor_diff = acceptor_scores_mut - acceptor_scores_ref
    donor_diff = donor_scores_mut - donor_scores_ref

    # Ensure differences are float64
    acceptor_diff = acceptor_diff.astype('float64')
    donor_diff = donor_diff.astype('float64')

    # Create DataFrames for score changes
    score_len = len(acceptor_diff)
    positions = range(score_len)
    bases = ['A', 'C', 'G', 'T']
    acceptor_score_change_df = pd.DataFrame(0.0, index=positions, columns=bases, dtype='float64')
    donor_score_change_df = pd.DataFrame(0.0, index=positions, columns=bases, dtype='float64')
    print(acceptor_score_change_df, donor_score_change_df)

    # Map mutation position to index in score differences
    # Assuming acceptor_diff and donor_diff align with the sequence positions
    # Mutation position in DataFrame is already 1-based
    for pos in positions:
        base = sequence[pos + flanking_size//2]
        acceptor_score_change_df.at[pos, base] = acceptor_diff[pos]
        donor_score_change_df.at[pos, base] = donor_diff[pos]
        
    print(acceptor_score_change_df, donor_score_change_df)

    # Save DataFrames to CSV files
    acceptor_score_change_df.to_csv(os.path.join(output_dir, 'acceptor_score_changes.csv'))
    donor_score_change_df.to_csv(os.path.join(output_dir, 'donor_score_changes.csv'))

    # Plot DNA logos
    generate_dna_logo(acceptor_score_change_df, os.path.join(output_dir, 'acceptor_score_difference_logo.png'), start=1000-5, end=1000+20)
    generate_dna_logo(donor_score_change_df, os.path.join(output_dir, 'donor_score_difference_logo.png'), start=1000-5, end=1000+20)

    # Plot the donor and acceptor sites
    plot_donor_acceptor_sites(acceptor_scores_ref, acceptor_scores_mut, donor_scores_ref, donor_scores_mut, output_dir)
    
def mutagenesis():
        
    model_types = ['pytorch', 'keras']
    flanking_sizes = [10000]
    exp_number = 1
    sample = 'mybpc3'
    mutation_position = 6000
    mutation_base = 'A'
    
    for model_type, flanking_size in itertools.product(model_types, flanking_sizes):
        if model_type == "keras":
            model_path = None
        elif model_type == "pytorch":
            model_path = f'/ccb/cybertron2/smao10/openspliceai/models/spliceai-mane/{flanking_size}nt/'
        else:
            print('not possible')
            exit(1)
        
        fasta_file = f'/ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/c1/data/{sample}.fa'
        
        # Initialize params
        output_dir = f"/ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/c1/results/exp_{exp_number}/{model_type}_{flanking_size}_{sample}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Load models (a list of models is passed)
        models, device = load_models(model_path, model_type, flanking_size)
        print(models)

        # Run the mutagenesis experiment
        exp_1(fasta_file, models, model_type, flanking_size, output_dir, device, mutation_position, mutation_base)

if __name__ == '__main__':
    mutagenesis()