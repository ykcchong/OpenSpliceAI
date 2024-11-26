import os
import tempfile
import mappy as mp
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import json

def read_and_split_txt_file(file_path):
    """
    Read sequences from a txt file and split into train and test datasets.
    """
    train_data = []
    test_data = []
    test_chromosomes = {'chr1', 'chr3', 'chr5', 'chr7', 'chr9'}
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                chrom = parts[0].split(':')[0]
                if chrom in test_chromosomes:
                    test_data.append((parts[0], parts[1]))
                else:
                    train_data.append((parts[0], parts[1]))
    
    return train_data, test_data

def identify_paralogs(train_data, test_data, min_identity, min_coverage, output_dir):
    """
    Identify paralogous sequences between train and test datasets using mappy.
    """
    print("Identifying paralogs...")
    
    # Create a temporary file with training sequences
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        for i, (_, seq) in enumerate(train_data):
            temp_file.write(f">seq{i}\n{seq}\n")
        temp_filename = temp_file.name

    # Create a mappy index from the training sequences
    try:
        aligner = mp.Aligner(temp_filename, preset="map-ont")
        if not aligner:
            raise Exception("Failed to load/build index")
    except Exception as e:
        print(f"Error creating mappy aligner: {str(e)}")
        os.unlink(temp_filename)
        return

    paralog_results = []

    for test_name, test_seq in tqdm(test_data):
        for hit in aligner.map(test_seq):
            identity = hit.mlen / hit.blen
            coverage = hit.blen / len(test_seq)
            paralog_results.append({
                "test_name": test_name,
                "identity": identity,
                "coverage": coverage
            })
            if identity >= min_identity and coverage >= min_coverage:
                break  # Count each test sequence only once

    os.unlink(temp_filename)

    # Write results to a JSON file
    with open(os.path.join(output_dir, 'paralog_results.json'), 'w') as f:
        json.dump(paralog_results, f)

    print(f"Paralog results saved to {os.path.join(output_dir, 'paralog_results.json')}")

def plot_distributions(input_file, output_dir):
    """
    Plot the distribution of sequence identity and coverage from a JSON file.
    """
    with open(input_file, 'r') as f:
        paralog_results = json.load(f)

    identities = [result['identity'] for result in paralog_results]
    coverages = [result['coverage'] for result in paralog_results]



    plt.figure(figsize=(5, 5))
    plt.scatter(identities, coverages, s=5, alpha=0.5)
    plt.title('Scatter plot of Sequence Identities & Coverages')
    plt.xlabel('Identities')
    plt.ylabel('Coverages')
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.hist(identities, bins=20, edgecolor='black')
    # plt.title('Distribution of Sequence Identity')
    # plt.xlabel('Identity')
    # plt.ylabel('Frequency')

    # plt.subplot(1, 2, 2)
    # plt.hist(coverages, bins=20, edgecolor='black')
    # plt.title('Distribution of Sequence Coverage')
    # plt.xlabel('Coverage')
    # plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'paralog_distributions.png'))
    plt.close()

    print(f"Distribution plots saved in {output_dir}")

def main(train_txt_path, output_dir, min_identity=0.8, min_coverage=0.8, visualize=True):
    os.makedirs(output_dir, exist_ok=True)

    # Read and split data from txt file
    print("Reading and splitting data...")
    train_data, test_data = read_and_split_txt_file(train_txt_path)
    print(f"Number of training sequences: {len(train_data)}")
    print(f"Number of testing sequences: {len(test_data)}")

    # # Identify paralogs and save results
    # identify_paralogs(train_data, test_data, min_identity, min_coverage, output_dir)

    # Read results and print statistics
    with open(os.path.join(output_dir, 'paralog_results.json'), 'r') as f:
        paralog_results = json.load(f)
    
    paralog_count = len(paralog_results)
    print(f"Number of paralogous sequences: {paralog_count}")
    print(f"Percentage of test sequences that are paralogs: {(paralog_count / len(test_data)) * 100:.2f}%")

    # Plot distributions if visualize is True
    if visualize:
        plot_distributions(os.path.join(output_dir, 'paralog_results.json'), output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split sequences, identify paralogs, and optionally plot distributions")
    parser.add_argument("--train_txt", type=str, help="Path to input txt file")
    parser.add_argument("--output_dir", type=str, help="Directory to save output files")
    parser.add_argument("--min_identity", type=float, default=0.8, help="Minimum identity for sequences to be considered paralogous")
    parser.add_argument("--min_coverage", type=float, default=0.8, help="Minimum coverage for sequences to be considered paralogous")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization plots")
    args = parser.parse_args()

    main(args.train_txt, args.output_dir, args.min_identity, args.min_coverage, args.visualize)