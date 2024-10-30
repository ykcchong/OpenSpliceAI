'''Replicating results from SpliceAI figure 1D'''

import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
from pyfaidx import Fasta
import logging
import platform
import os, glob
import torch
from openspliceai.train_base.spliceai import SpliceAI
from spliceaitoolkit.constants import *
import logomaker
import matplotlib.pyplot as plt

import itertools
from keras import backend as K
    
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
        model_files = glob.glob(os.path.join(model_path, '*.p[th]')) # gets all PyTorch models from supplied directory
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
    from tensorflow import keras
    
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
            from tensorflow import keras
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
    # pad_size = [flanking_size // 2, flanking_size // 2]
    # x = 'N' * pad_size[0] + seq + 'N' * pad_size[1]
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

def predict_pytorch(models, flanking_size, seq, strand='+', device='cuda'):
    # Prepare the sequence with padding
    # pad_size = [flanking_size // 2, flanking_size // 2]
    # x = 'N' * pad_size[0] + seq + 'N' * pad_size[1]
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

# Function to mutate a base to the three other bases
def mutate_base(base):
    bases = ['A', 'C', 'G', 'T']
    return [b for b in bases if b != base]

# Function to calculate average score change
def calculate_average_score_change(ref_scores, mut_scores):
    return ref_scores - np.mean(mut_scores, axis=0)

# Function to generate DNA logo
def generate_dna_logo(score_changes, output_file):
    
    data_df = pd.DataFrame(score_changes, columns=['A', 'C', 'G', 'T']).astype(float)
    # Fill any missing values with 0, just in case
    data_df = data_df.fillna(0)
    # data_df = logomaker.transform_matrix(pd.DataFrame(data_df), from_type='probability', to_type='weight')
    logo = logomaker.Logo(data_df)
    logo.ax.set_title('DNA Logo - Score Change by Base')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.show()

# Function to generate line plot for average score change
def plot_average_dna_logo(average_score_change, sequence, output_file):
    # Create a DataFrame where the index is the position, and columns are the nucleotides
    data = []
    for i, base in enumerate(sequence):
        row = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
        row[base] = average_score_change[i]  # Assign the score to the correct base
        data.append(row)
    
    # Convert to a DataFrame
    df = pd.DataFrame(data)
    # df = logomaker.transform_matrix(pd.DataFrame(df), from_type='probability', to_type='weight')
    
    # Create a DNA logo plot using Logomaker
    nn_logo = logomaker.Logo(df, figsize=(30, 3))
    
    ### Customize the plot
    nn_logo.style_spines(visible=False)
    nn_logo.style_spines(spines=['left', 'bottom'], visible=True)
    nn_logo.ax.set_ylabel("Average Score Change")
    nn_logo.ax.set_xlabel("Position")
    
    # Save the plot to the output file
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.show()

##############################################
## MUTAGENESIS EXPERIMENT
##############################################

# Main function for mutagenesis experiment
def exp_4(fasta_file, models, model_type, flanking_size, output_dir, device, site):
        
    # Load fasta file and sequence
    fasta = Fasta(fasta_file)
    sequence = str(fasta[0])
    offset = flanking_size // 2
    seq_length = len(sequence) - flanking_size
    
    # Crop sequence 
    if flanking_size != 10000:
        difference = 5000 - offset
        sequence = sequence[difference:-difference]    
    
    # Run the prediction
    ref_acceptor_scores, ref_donor_scores = predict(models, model_type, flanking_size, sequence, device=device)
    assert(len(ref_acceptor_scores) == len(ref_donor_scores) == seq_length)
    
    # Convert reference scores to pandas DataFrame
    acceptor_df = pd.DataFrame(ref_acceptor_scores, columns=['ref_acceptor_score'])
    donor_df = pd.DataFrame(ref_donor_scores, columns=['ref_donor_score'])

    acceptor_df.to_csv(f'{output_dir}/acceptor.csv', index=False)
    values = acceptor_df['ref_acceptor_score'].to_list()
    print(sequence[offset:-offset])
    print(values)
    plot_average_dna_logo(values, sequence[offset:-offset], f'{output_dir}/acceptor.png')
    
    donor_df.to_csv(f'{output_dir}/donor.csv', index=False)
    values = donor_df['ref_donor_score'].to_list()
    plot_average_dna_logo(values, sequence[offset:-offset], f'{output_dir}/donor.png')

def mutagenesis():
    '''
    Examine whether model can capture AT-AC non-canonical splice sites: 
    
    '''
    
    model_types = ['pytorch', 'keras']
    # sites = ['acceptor']
    # flanking_sizes = [80, 400, 2000, 10000]
    site = 'acceptor'
    flanking_sizes = [10000]
    exp_number = 1
    sample_number = 6
    just_visualize = False
    orig_style = False   
    
    for model_type, flanking_size in itertools.product(model_types, flanking_sizes):
        if model_type == "keras":
            model_path = None
        elif model_type == "pytorch":
            model_path = f'/ccb/cybertron/smao10/openspliceai/models/spliceai-mane/{flanking_size}nt/model_{flanking_size}nt_rs14.pth'
        else:
            print('not possible')
            exit(1)
            
        fasta_file = f'/ccb/cybertron/smao10/openspliceai/experiments/mutagenesis/experiment_4/data/{site}_{sample_number}.fa'
        
        # Initialize params
        output_dir = f"/ccb/cybertron/smao10/openspliceai/experiments/mutagenesis/experiment_4/results/exp_{exp_number}/{model_type}_{flanking_size}_{site}_samp{sample_number}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # if just_visualize:
        #     score_change_files = [f'{output_dir}/acceptor_score_change.csv', f'{output_dir}/donor_score_change.csv']
        #     average_change_files = [f'{output_dir}/acceptor_average_score_change.csv', f'{output_dir}/donor_average_score_change.csv']
        #     fasta = Fasta(fasta_file)
        #     sequence = str(fasta[0])
            
        #     for f in score_change_files:
        #         if os.path.exists(f):
        #             df = pd.read_csv(f)
        #             print(df)
        #             output = f.replace('.csv', '.png')
        #             print(output)
        #             generate_dna_logo(df, output)
            
        #     for f in average_change_files:
        #         if os.path.exists(f):
        #             df = pd.read_csv(f)
        #             values = df['0'].to_list()
        #             print(values)
        #             output = f.replace('.csv', '.png')
        #             print(output)
        #             plot_average_dna_logo(values, sequence[5000:-5000], output, orig_style)
        
        #     continue
            
        # Load models (a list of models is passed)
        models, device = load_models(model_path, model_type, flanking_size)

        # Run the mutagenesis experiment
        exp_4(fasta_file, models, model_type, flanking_size, output_dir, device, site)

if __name__ == '__main__':
    
    mutagenesis()