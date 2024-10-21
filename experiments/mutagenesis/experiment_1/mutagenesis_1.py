import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
from pyfaidx import Fasta
import logging
import platform
import sys, os, glob
import torch
from spliceaitoolkit.predict.spliceai import SpliceAI
from spliceaitoolkit.constants import *
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
        model_files = glob.glob(os.path.join(model_path, '*.pt')) # gets all PyTorch models from supplied directory
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
            logging.error(f"No valid Keras models found in directory: {model_path}")
            exit()
                
        return models, None
    
    elif os.path.isfile(model_path): # file supplied
        try:
            return [keras.models.load_model(model_path)], None
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

def predict_keras(models, flanking_size, seq, strand='+'):
    # Prepare the sequence with padding
    pad_size = [flanking_size // 2, flanking_size // 2]
    x = 'N' * pad_size[0] + seq + 'N' * pad_size[1]

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

def predict_pytorch(models, flanking_size, seq, strand='+', device='cuda'):
    # Prepare the sequence with padding
    pad_size = [flanking_size // 2, flanking_size // 2]
    x = 'N' * pad_size[0] + seq + 'N' * pad_size[1]

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

def predict(models, model_type, flanking_size, seq, strand='+', device='cuda'):
    if model_type == 'keras':
        return predict_keras(models, flanking_size, seq, strand)
    elif model_type == 'pytorch':
        return predict_pytorch(models, flanking_size, seq, strand, device)
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

    Args:
        score_changes_df (pd.DataFrame): DataFrame with positions as index and columns ['A', 'C', 'G', 'T']
        output_file (str): Path to save the DNA logo image.
        start (int, optional): Start position for plotting.
        end (int, optional): End position for plotting.
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
    logo = logomaker.Logo(data_df, shade_below=0, fade_below=0, font_name='Arial Rounded MT Bold')
    logo.ax.set_ylabel('Score Change')
    logo.ax.set_xlabel('Position')
    plt.savefig(output_file)
    plt.close()

##############################################
## MUTAGENESIS EXPERIMENT
##############################################

def exp_1(fasta_file, models, model_type, flanking_size, output_dir, device, scoring_position_global, mutation_position_global, mutation_length, site):
    '''
    Mutate a single base -> measure PWM change over all bases.
    '''
    # Load fasta file
    fasta = Fasta(fasta_file)
    header = list(fasta.keys())[0]
    sequence = str(fasta[header])
    
    # Extract the site position from the header
    # Example header: '>chr1:10000-10800(+)_donor_site_10400'
    header_parts = header.split('_')
    site_info = header_parts[-1]  # '10400'
    selected_site_position = int(site_info)
    
    # Calculate the relative position of the site in the sequence
    sequence_length = len(sequence)
    window_size = sequence_length
    half_window = window_size // 2
    scoring_position = half_window  # Center of the sequence
    
    # Now adjust mutation_position based on the scoring_position
    mutation_position = scoring_position
    
    bases_to_mutate = sequence[mutation_position:mutation_position+mutation_length]
    bases_to_mutate = tuple(bases_to_mutate)
    possible_mutations = get_mutations(bases_to_mutate)
    
    # Get the prediction of the base sequence first
    acceptor_scores_ref, donor_scores_ref = predict(models, model_type, flanking_size, sequence, device=device)
    
    # Initialize dictionaries to store score changes
    acceptor_score_changes = {}
    donor_score_changes = {}
    
    ref_base = sequence[mutation_position]
    
    # Iterate over all possible mutations
    for mutation in possible_mutations:
        mutated_base = mutation[0]  # Since mutation is a tuple of length 1
        mutated_sequence = list(sequence)
        # Replace the base at mutation_position with the mutated base
        mutated_sequence[mutation_position] = mutated_base
        mutated_sequence = ''.join(mutated_sequence)
        
        # Predict scores for mutated sequence
        acceptor_scores_mut, donor_scores_mut = predict(models, model_type, flanking_size, mutated_sequence, device=device)
        
        # Compute score differences
        acceptor_diff = acceptor_scores_mut - acceptor_scores_ref
        donor_diff = donor_scores_mut - donor_scores_ref
        
        # Store the score differences
        acceptor_score_changes[mutated_base] = acceptor_diff
        donor_score_changes[mutated_base] = donor_diff
    
    # Create DataFrames for score changes
    positions = range(len(sequence))
    bases = ['A', 'C', 'G', 'T']
    acceptor_score_change_df = pd.DataFrame(0.0, index=positions, columns=bases)
    donor_score_change_df = pd.DataFrame(0.0, index=positions, columns=bases)
    
    for base in bases:
        if base != ref_base:
            acceptor_score_change_df[base] = acceptor_score_changes.get(base, 0)
            donor_score_change_df[base] = donor_score_changes.get(base, 0)
        else:
            # The score change when mutating to the same base is zero
            acceptor_score_change_df[base] = 0.0
            donor_score_change_df[base] = 0.0
    
    # Write all DataFrames to CSV files
    acceptor_score_change_df.to_csv(os.path.join(output_dir, 'acceptor_score_changes.csv'))
    donor_score_change_df.to_csv(os.path.join(output_dir, 'donor_score_changes.csv'))
    
    # Plot the DNA logos
    generate_dna_logo(acceptor_score_change_df, os.path.join(output_dir, 'acceptor_dna_logo.png'))
    generate_dna_logo(donor_score_change_df, os.path.join(output_dir, 'donor_dna_logo.png'))

def mutagenesis():
        
    model_types = ['pytorch', 'keras']
    scoring_positions = {'donor': 198, 'acceptor': 201}
    flanking_sizes = [80, 400, 2000, 10000]
    exp_number = 1
    sample_number = 1
    site = 'donor'
    scoring_position = scoring_positions[site]
    mutation_position = scoring_position  # Assuming mutation position is the scoring position
    mutation_length = 2 # Mutate a single base
    
    for model_type, flanking_size in itertools.product(model_types, flanking_sizes):
        if model_type == "keras":
            model_path = None
        elif model_type == "pytorch":
            model_path = f'/ccb/cybertron/smao10/openspliceai/models/spliceai-mane/{flanking_size}nt/model_{flanking_size}nt_rs42.pt'
        else:
            print('not possible')
            exit(1)
        
        fasta_file = f'/ccb/cybertron/smao10/openspliceai/experiments/mutagenesis/experiment_1/data/{site}_{sample_number}.fa'
        
        # Initialize params
        output_dir = f"/ccb/cybertron/smao10/openspliceai/experiments/mutagenesis/experiment_1/results/exp_{exp_number}/{model_type}_{flanking_size}_{site}_samp{sample_number}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Load models (a list of models is passed)
        models, device = load_models(model_path, model_type, flanking_size)

        # Run the mutagenesis experiment
        exp_1(fasta_file, models, model_type, flanking_size, output_dir, device, scoring_position, mutation_position, mutation_length, site)

if __name__ == '__main__':
    mutagenesis()