import h5py
import numpy as np
from tqdm import tqdm
import time
import argparse
from openspliceai.constants import *
from openspliceai.create_data.utils import ceil_div, replace_non_acgt_to_n, create_datapoints

CHUNK_SIZE = 100  # size of chunks to process data in

def convert_to_dataset(input_file, output_file):
    print(f"Converting {input_file} to dataset format...")
    start_time = time.process_time()

    with h5py.File(input_file, 'r') as h5f:
        SEQ = h5f['SEQ'][:]
        LABEL = h5f['LABEL'][:]

    with h5py.File(output_file, 'w') as h5f2:
        seq_num = len(SEQ)
        num_chunks = ceil_div(seq_num, CHUNK_SIZE)
        
        for i in tqdm(range(num_chunks), desc='Processing chunks...'):
            # each dataset has CHUNK_SIZE genes
            if i == num_chunks - 1:  # if last chunk, process remainder or full chunk size if no remainder
                NEW_CHUNK_SIZE = seq_num % CHUNK_SIZE or CHUNK_SIZE 
            else:
                NEW_CHUNK_SIZE = CHUNK_SIZE
            
            X_batch, Y_batch = [], [[]]
            for j in range(NEW_CHUNK_SIZE):
                idx = i*CHUNK_SIZE + j
                seq_decode = SEQ[idx].decode('utf-8')
                label_decode = LABEL[idx].decode('utf-8')
                fixed_seq = replace_non_acgt_to_n(seq_decode)
                X, Y = create_datapoints(fixed_seq, label_decode)
                X_batch.extend(X)
                Y_batch[0].extend(Y[0])
            
            # Convert batches to arrays and save as HDF5
            X_batch = np.asarray(X_batch).astype('int8')
            Y_batch[0] = np.asarray(Y_batch[0]).astype('int8')
            print("X_batch shape:", X_batch.shape)
            print("Y_batch shape:", Y_batch[0].shape)

            h5f2.create_dataset(f'X{i}', data=X_batch)
            h5f2.create_dataset(f'Y{i}', data=Y_batch)

    print(f"Conversion completed in {time.process_time() - start_time} seconds")

def main(args):
    input_file = args.input_file
    output_file = args.output_file

    convert_to_dataset(input_file, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert filtered h5 file to dataset.h5 format")
    parser.add_argument("--input_file", type=str, help="Path to the input filtered h5 file")
    parser.add_argument("--output_file", type=str, help="Path to save the output dataset.h5 file")
    args = parser.parse_args()

    main(args)