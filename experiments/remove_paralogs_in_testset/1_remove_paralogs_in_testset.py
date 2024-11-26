import os
import tempfile
import h5py
import numpy as np
import mappy as mp
import logging
from tqdm import tqdm
from Bio import SeqIO

def read_txt_file(file_path):
    """
    Read sequences from a txt file.
    """
    data = []
    excluded_chromosomes = {'chr1', 'chr3', 'chr5', 'chr7', 'chr9'}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                chrom = parts[0].split(':')[0]
                if chrom not in excluded_chromosomes:
                    data.append((parts[0], parts[1]))
    return data

def read_h5_file(file_path):
    """
    Read data from an h5 file.
    """
    with h5py.File(file_path, 'r') as h5f:
        data = {
            'NAME': h5f['NAME'][:],
            'CHROM': h5f['CHROM'][:],
            'STRAND': h5f['STRAND'][:],
            'TX_START': h5f['TX_START'][:],
            'TX_END': h5f['TX_END'][:],
            'SEQ': h5f['SEQ'][:],
            'LABEL': h5f['LABEL'][:]
        }
    return data

def read_fasta_file(file_path):
    """
    Read sequences from a FASTA file.
    """
    data = []
    for record in SeqIO.parse(file_path, "fasta"):
        data.append((record.id, str(record.seq)))
    return data

def write_h5_file(output_path, data):
    """
    Write the data to an h5 file.
    """
    with h5py.File(output_path, 'w') as h5f:
        dt = h5py.string_dtype(encoding='utf-8')
        for key, value in data.items():
            h5f.create_dataset(key, data=np.asarray(value, dtype=dt), dtype=dt)

def remove_paralogous_sequences(train_data, test_data, min_identity, min_coverage, output_dir):
    """
    Remove paralogous sequences between train and test datasets using mappy.
    """
    print(f"Starting paralogy removal process...")
    print(f"Initial train set size: {len(train_data)}")
    print(f"Initial test set size: {len(test_data['SEQ'])}")

    # Create a temporary file with training sequences
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        for i, (_, seq) in enumerate(train_data):
            temp_file.write(f">seq{i}\n{seq}\n")
        temp_filename = temp_file.name

    # Create a mappy index from the training sequences
    print("Creating mappy index from training sequences...")
    try:
        aligner = mp.Aligner(temp_filename, preset="map-ont")
        if not aligner:
            raise Exception("Failed to load/build index")
    except Exception as e:
        logging.error(f"Error creating mappy aligner: {str(e)}")
        os.unlink(temp_filename)
        return test_data

    filtered_test_data = {key: [] for key in test_data.keys()}
    paralogous_count = 0
    total_count = len(test_data['SEQ'])

    with open(f"{output_dir}/removed_paralogs.txt", "w") as fw:
        print("Starting to process test sequences...")
        for i in tqdm(range(total_count)):
            test_seq = test_data['SEQ'][i].decode('utf-8')
            is_paralogous = False
            for hit in aligner.map(test_seq):
                identity = hit.mlen / hit.blen
                coverage = hit.blen / len(test_seq)
                if identity >= min_identity and coverage >= min_coverage:
                    fw.write(f"{test_data['NAME'][i].decode('utf-8')}\t{identity}\t{coverage}\n")
                    is_paralogous = True
                    paralogous_count += 1
                    break
            if not is_paralogous:
                for key in filtered_test_data.keys():
                    filtered_test_data[key].append(test_data[key][i])

    print(f"Paralogy removal process completed.")
    print(f"Number of paralogous sequences removed: {paralogous_count}")
    print(f"Final test set size: {len(filtered_test_data['SEQ'])}")
    print(f"Percentage of test set removed: {(paralogous_count / total_count) * 100:.2f}%")

    os.unlink(temp_filename)
    return filtered_test_data

def main(spliceai_keras_train_txt_path, openspliceai_train_fasta_path, test_h5_path, output_dir, min_identity=0.8, min_coverage=0.8):
    os.makedirs(output_dir, exist_ok=True)

    # Read data from txt file for training sequences
    print("Reading training data from txt file...")
    train_data = read_txt_file(spliceai_keras_train_txt_path)
    print(f"Number of training sequences after filtering: {len(train_data)}")

    # Read additional training data from FASTA file
    if openspliceai_train_fasta_path:
        print("Reading additional training data from FASTA file...")
        fasta_data = read_fasta_file(openspliceai_train_fasta_path)
        print(f"Number of sequences from FASTA file: {len(fasta_data)}")
        train_data.extend(fasta_data)

    # Read testing data from h5 file
    print("Reading testing data from h5 file...")
    test_data = read_h5_file(test_h5_path)

    # Remove paralogs
    filtered_test_data = remove_paralogous_sequences(train_data, test_data, min_identity, min_coverage, output_dir)

    # Write filtered test data to new h5 file
    output_h5_path = os.path.join(output_dir, 'filtered_test_data.h5')
    write_h5_file(output_h5_path, filtered_test_data)
    print(f"Filtered test data written to {output_h5_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Remove paralogs from test dataset given train txt and test h5 files")
    parser.add_argument("--spliceai_keras_train_txt", type=str, help="Path to training data txt file")
    parser.add_argument("--test_h5", type=str, help="Path to testing data h5 file")
    parser.add_argument("--openspliceai_train_fasta", type=str, help="Path to additional training data in FASTA format", default=None)
    parser.add_argument("--output_dir", type=str, help="Directory to save output files")
    parser.add_argument("--min_identity", type=float, default=0.8, help="Minimum identity for sequences to be considered paralogous")
    parser.add_argument("--min_coverage", type=float, default=0.8, help="Minimum coverage for sequences to be considered paralogous")
    args = parser.parse_args()

    main(args.spliceai_keras_train_txt, args.openspliceai_train_fasta, args.test_h5, args.output_dir, args.min_identity, args.min_coverage)
