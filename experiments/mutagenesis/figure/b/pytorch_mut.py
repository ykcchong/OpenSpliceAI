import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
from pyfaidx import Fasta
import logging
import platform
import os, glob
import torch
# from spliceaitoolkit.predict.spliceai import SpliceAI
from openspliceai.train_base.spliceai import SpliceAI
from vis import mutate_base, calculate_average_score_change, generate_dna_logo, plot_average_score_change
from spliceaitoolkit.constants import *

import sys
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
## MUTAGENESIS EXPERIMENT
##############################################

# Main function for mutagenesis experiment
def exp_2(fasta_file, models, model_type, flanking_size, output_dir, device, scoring_position, site, max_seq_length=400, debug=False):
    '''
    Mutate every base in 400nt window around donor/acceptor site and measure the score change at the donor/acceptor site.
    '''
    # Load fasta file
    sequences = Fasta(fasta_file)
    
    # Initialize DataFrames to store cumulative sums and counts
    cumulative_acceptor_df = pd.DataFrame(0, index=range(max_seq_length), columns=['ref', 'A', 'C', 'G', 'T'], dtype='float64')
    cumulative_donor_df = pd.DataFrame(0, index=range(max_seq_length), columns=['ref', 'A', 'C', 'G', 'T'], dtype='float64')

    count_df = pd.DataFrame(0, index=range(max_seq_length), columns=['ref', 'A', 'C', 'G', 'T'])  # Store counts for averaging

    # Iterate over each transcript
    all_sequences = list(sequences.keys())[:100] # TODO: locked to 100 sequences for testing
    for i in range(len(all_sequences)):
        seq_id = all_sequences[i]
        sequence = str(sequences[seq_id])
        
        # Crop sequence 
        crop = (10000 - flanking_size) // 2
        if crop > 0:
            sequence = sequence[crop:-crop]
        seq_length = len(sequence) - flanking_size
        
        if debug:
            print(f"***** Processing sequence {i}/{len(all_sequences)}: {seq_id}, len {seq_length} *****", file=sys.stderr)

        # Iterate over each base in the transcript
        for raw_pos in tqdm(range(seq_length)):
            
            pos = raw_pos + (flanking_size // 2) 
                        
            ref_base = sequence[pos]
            mutations = mutate_base(ref_base)

            # Create placeholder arrays to store scores, with an extra row for 'ref'
            acceptor_scores = np.zeros(5)  # for ref, A, C, G, T
            donor_scores = np.zeros(5)  # for ref, A, C, G, T
            
            # Get reference sequence scores
            ref_sequence = sequence[:pos] + ref_base + sequence[pos + 1:]
            ref_acceptor_scores, ref_donor_scores = predict(models, model_type, flanking_size, ref_sequence, device=device)
            
            # Extract the reference scores
            ref_acceptor_score = ref_acceptor_scores[scoring_position]
            ref_donor_score = ref_donor_scores[scoring_position]
            
            if debug:
                acceptor_max_position = np.argmax(acceptor_scores)
                donor_max_position = np.argmax(donor_scores)
                print(f'mutating {pos} -- acceptor_max_position: {acceptor_max_position}, donor_max_position: {donor_max_position}', file=sys.stderr)
            
            acceptor_scores[0] = ref_acceptor_score
            donor_scores[0] = ref_donor_score

            # Store the reference score in the corresponding base column as well
            base_order = ['A', 'C', 'G', 'T']
            if ref_base in base_order:
                acceptor_scores[base_order.index(ref_base) + 1] = acceptor_scores[0]
                donor_scores[base_order.index(ref_base) + 1] = donor_scores[0]

            # Mutate the base and get scores for each mutation
            for i, mut_base in enumerate(mutations):
                mut_sequence = sequence[:pos] + mut_base + sequence[pos + 1:]
                
                # Predict the scores for the mutated sequence
                mut_acceptor_scores, mut_donor_scores = predict(models, model_type, flanking_size, mut_sequence, device=device)
                mut_acceptor_score = mut_acceptor_scores[scoring_position]
                mut_donor_score = mut_donor_scores[scoring_position]
                    
                acceptor_scores[base_order.index(mut_base) + 1] = mut_acceptor_score
                donor_scores[base_order.index(mut_base) + 1] = mut_donor_score
                
            if debug:
                if site == 'acceptor':
                    print(f"Position: {pos}, Acceptor Scores: {' '.join(map(str, acceptor_scores))}", file=sys.stderr)
                else:
                    print(f"Position: {pos}, Donor Scores: {' '.join(map(str, donor_scores))}", file=sys.stderr)
            
            # Update cumulative sums and counts
            cumulative_acceptor_df.loc[raw_pos, ['ref', 'A', 'C', 'G', 'T']] += acceptor_scores
            cumulative_donor_df.loc[raw_pos, ['ref', 'A', 'C', 'G', 'T']] += donor_scores

            count_df.loc[raw_pos, ['ref', 'A', 'C', 'G', 'T']] += 1

            # release memory if possible
            if model_type == 'keras':
                K.clear_session()  # clear the session after each prediction
    
    if debug:          
        print("***** Finished processing all sequences *****", file=sys.stderr)
        print(cumulative_acceptor_df, file=sys.stderr)
        print(count_df, file=sys.stderr)
    
    # Calculate the rolling average across all sequences
    acceptor_avg_df = cumulative_acceptor_df / count_df
    donor_avg_df = cumulative_donor_df / count_df

    ### GENERATE PLOTS ###

    # Generate DNA logos for acceptor and donor score changes
    acceptor_score_change_df = acceptor_avg_df.apply(
            lambda row: pd.Series({i: row['ref'] - row[i] for i in ['A', 'C', 'G', 'T']}),
            axis=1
        )
    donor_score_change_df = donor_avg_df.apply(
            lambda row: pd.Series({i: row['ref'] - row[i] for i in ['A', 'C', 'G', 'T']}),
            axis=1
        )

    if site == 'acceptor':
        acceptor_score_change_df.to_csv(f'{output_dir}/acceptor_dna_logo.csv', index=False)
        generate_dna_logo(acceptor_score_change_df, site, f'{output_dir}/acceptor_dna_logo.png')
    else:
        donor_score_change_df.to_csv(f'{output_dir}/donor_dna_logo.csv', index=False)
        generate_dna_logo(donor_score_change_df, site, f'{output_dir}/donor_dna_logo.png')

    # Calculate average score change for each base and plot
    acceptor_score_change = acceptor_avg_df.apply(lambda row: row['ref'] - np.mean([row['A'], row['C'], row['G'], row['T']]), axis=1)
    donor_score_change = donor_avg_df.apply(lambda row: row['ref'] - np.mean([row['A'], row['C'], row['G'], row['T']]), axis=1)

    if site == 'acceptor':
        acceptor_score_change.to_csv(f'{output_dir}/acceptor_average_score_change.csv', index=False)
        plot_average_score_change(acceptor_score_change, f'{output_dir}/acceptor_average_score_change.png')
    else:
        donor_score_change.to_csv(f'{output_dir}/donor_average_score_change.csv', index=False)
        plot_average_score_change(donor_score_change, f'{output_dir}/donor_average_score_change.png')

    ### WRITE SCORES TO FILE ###

    # Add prefixes to differentiate between acceptor and donor columns
    acceptor_combined_df = pd.concat([acceptor_avg_df, acceptor_score_change_df.add_suffix('_change')], axis=1)
    acceptor_combined_df = acceptor_combined_df.add_prefix('acceptor_')

    donor_combined_df = pd.concat([donor_avg_df, donor_score_change_df.add_suffix('_change')], axis=1)
    donor_combined_df = donor_combined_df.add_prefix('donor_')

    # Concatenate acceptor and donor DataFrames
    combined_df = pd.concat([acceptor_combined_df, donor_combined_df], axis=1)

    # Save everything to a single CSV file
    combined_df.to_csv(f'{output_dir}/scores.csv', index=False)


def mutagenesis():
        
    model_types = ['pytorch']
    # model_types = ['keras', 'pytorch']
    sites = ['donor', 'acceptor']
    scoring_positions = {'donor': 198, 'acceptor': 201}
    # flanking_sizes = [80, 400, 2000, 10000]
    flanking_sizes = [10000]
    sample_number = 7
    debug = False

    visualize = False
    
    for model_type, flanking_size, site in itertools.product(model_types, flanking_sizes, sites):
        if model_type == "keras":
            model_path = None
        elif model_type == "pytorch":
            model_path = f'/ccb/cybertron2/smao10/openspliceai/models/spliceai-mane/{flanking_size}nt/'
        else:
            print('not possible') 
            exit(1)
        
        scoring_position = scoring_positions[site]
        
        fasta_file = f'/ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/b/data/{site}_{sample_number}.fa'
        
        # Initialize params
        output_dir = f"/ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/b/{model_type}_{flanking_size}_{site}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        if visualize:
            df = pd.read_csv(f'{output_dir}/{site}_dna_logo.csv')
            generate_dna_logo(df, site, f'{output_dir}/{site}_dna_logo.png')
            continue

        # Load models (a list of models is passed)
        models, device = load_models(model_path, model_type, flanking_size)
        
        # Run the mutagenesis experiment
        exp_2(fasta_file, models, model_type, flanking_size, output_dir, device, scoring_position, site, debug=debug)

if __name__ == '__main__':
    
    mutagenesis()