import argparse
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
from pyfaidx import Fasta
import logging
import platform
import os, glob
import torch
from tensorflow import keras
import itertools
from keras import backend as K
    
##############################################
## LOADING PYTORCH AND KERAS MODELS
##############################################

def setup_device():
        """Select computation device based on availability."""
        device_str = "cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" else "cpu"
        return torch.device(device_str)

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
    if model_type == "keras":
    
        if model_path == None:
            paths = (f'models/SpliceAI/SpliceNet{CL}_c{x}.h5' for x in range(1, 6))
            models = [load_keras_models(x) for x in paths]
            return models, None
        return load_keras_models(model_path)
    else:
        logging.error("Invalid model type specified")
        exit()

##############################################
## UTILS: ONE-HOT ENCODING, MUTATION, LOGOS
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
    map = np.asarray([[0, 0, 0, 0],  # N or any invalid character
                      [1, 0, 0, 0],  # A
                      [0, 1, 0, 0],  # C
                      [0, 0, 1, 0],  # G
                      [0, 0, 0, 1]]) # T

    # Replace nucleotides with corresponding indices
    seq = seq.upper().replace('A', '\x01').replace('C', '\x02')
    seq = seq.replace('G', '\x03').replace('T', '\x04').replace('N', '\x00')

    # Convert the sequence to one-hot encoded numpy array
    return map[np.fromstring(seq, np.int8) % 5]

# Function to mutate a base to the three other bases
def mutate_base(base):
    bases = ['A', 'C', 'G', 'T']
    return [b for b in bases if b != base]

# Function to calculate average score change
def calculate_average_score_change(ref_scores, mut_scores):
    return ref_scores - np.mean(mut_scores, axis=0)
    
    
##############################################
## PREDICT FUNCTIONS
##############################################

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

def predict(models, model_type, flanking_size, seq, strand='+', device='cuda'):
    if model_type == 'keras':
        return predict_keras(models, flanking_size, seq, strand)
    else:
        logging.error("Invalid model type")
        exit()
    
##############################################
## MUTAGENESIS EXPERIMENT
##############################################

# Main function for mutagenesis experiment
def exp_2(fasta_file, models, model_type, flanking_size, output_dir, device, scoring_position, site, max_seq_length=400):
    '''Mutate all bases, score donor/acceptor site.'''
    # Load fasta file
    sequences = Fasta(fasta_file)
    
    # Initialize list to store results
    results = []
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate over each transcript
    for seq_id in sequences.keys():
        sequence = str(sequences[seq_id])
        seq_length = len(sequence)
    
        # Iterate over each base in the transcript
        for pos in tqdm(range(seq_length), desc=f'Processing {seq_id}'):
            ref_base = sequence[pos]
            mutations = mutate_base(ref_base)
    
            # Create placeholder arrays to store scores, with an extra slot for 'ref'
            acceptor_scores = np.zeros(5)  # for ref, A, C, G, T
            donor_scores = np.zeros(5)     # for ref, A, C, G, T
            
            # Get reference sequence scores
            ref_sequence = sequence
            ref_acceptor_scores, ref_donor_scores = predict(models, model_type, flanking_size, ref_sequence, device=device)
            
            # Extract the reference scores
            ref_acceptor_score = ref_acceptor_scores[scoring_position]
            ref_donor_score = ref_donor_scores[scoring_position]
            
            acceptor_scores[0] = ref_acceptor_score
            donor_scores[0] = ref_donor_score
    
            # Store the reference score in the corresponding base column as well
            base_order = ['A', 'C', 'G', 'T']
            if ref_base in base_order:
                idx = base_order.index(ref_base) + 1
                acceptor_scores[idx] = ref_acceptor_score
                donor_scores[idx] = ref_donor_score
    
            # Mutate the base and get scores for each mutation
            for mut_base in mutations:
                mut_sequence = sequence[:pos] + mut_base + sequence[pos + 1:]
                
                # Predict the scores for the mutated sequence
                mut_acceptor_scores, mut_donor_scores = predict(models, model_type, flanking_size, mut_sequence, device=device)
                mut_acceptor_score = mut_acceptor_scores[scoring_position]
                mut_donor_score = mut_donor_scores[scoring_position]
                
                idx = base_order.index(mut_base) + 1    
                acceptor_scores[idx] = mut_acceptor_score
                donor_scores[idx] = mut_donor_score
    
            # Store the results
            bases = ['ref'] + base_order
            for i, base in enumerate(bases):
                results.append({
                    'seq_id': seq_id,
                    'position': pos,
                    'ref_base': ref_base,
                    'mut_base': base,
                    'acceptor_score': acceptor_scores[i],
                    'donor_score': donor_scores[i]
                })
    
            # Release memory if possible
            if model_type == 'keras':
                K.clear_session()  # Clear the session after each prediction
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to a single CSV file
    output_file = os.path.join(output_dir, 'mutagenesis_results.csv')
    results_df.to_csv(output_file, index=False)
    

def mutagenesis(batch_num, flanking_size):
        
    model_type = 'keras'
    sites = ['donor', 'acceptor']
    scoring_positions = {'donor': 198, 'acceptor': 201}
    
    for site in sites:
        
        scoring_position = scoring_positions[site]
            
        fasta_file = f'experiments/mutagenesis/experiment_2/data/keras_job/{site}_batch{batch_num}.fa'
        
        # Initialize params
        output_dir = f"experiments/mutagenesis/experiment_2/results/keras_job/{model_type}_{flanking_size}_{site}_{batch_num}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Load models (a list of models is passed)
        models, device = load_models(None, model_type, flanking_size)

        # Run the mutagenesis experiment
        exp_2(fasta_file, models, model_type, flanking_size, output_dir, device, scoring_position, site)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_num", type=int, help="Batch number")
    parser.add_argument("model_size", type=int, help="Model size")
    args = parser.parse_args()
    
    mutagenesis(args.batch_num, args.model_size)
