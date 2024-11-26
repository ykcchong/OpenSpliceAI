import argparse
import os, sys
import numpy as np
import platform
import time
import matplotlib.pyplot as plt
import sys

RANDOM_SEED = 42

def initialize_paths(output_dir, flanking_size, sequence_length, rs, rs_idx, species, experiment, target):
    """Initialize project directories and create them if they don't exist."""
    # Experiment paths
    if experiment == "my_split_paralog_removal":
        res_root = f"/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/train_my_split_paralog_removal_outdir/{species}/flanking_{flanking_size}/SpliceAI_{species}_train_{flanking_size}_{rs_idx}_rs{rs}/{rs_idx}/"
    elif experiment == "my_split_no_paralog_removal":
        res_root = f"/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/train_my_split_no_paralog_removal_outdir/{species}/flanking_{flanking_size}/SpliceAI_{species}_train_{flanking_size}_{rs_idx}_rs{rs}/{rs_idx}/"
    elif experiment == "spliceai_default_no_paralog_removal":
        res_root = f"/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/train_spliceai_default_no_paralog_removal_outdir/{species}/flanking_{flanking_size}/SpliceAI_{species}_train_{flanking_size}_{rs_idx}_rs{rs}/{rs_idx}/"
    elif experiment == "new_model_arch_spliceai_default_paralog_removed":
        res_root = f"/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/train_outdir/new_model_arch_spliceai_default_paralog_removed/{species}/flanking_{flanking_size}/SpliceAI_{species}_train_{flanking_size}_{rs_idx}_rs{rs}/{rs_idx}/"
    elif experiment == "SpliceAI-keras_data":
        res_root = f"/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/train_outdir/SpliceAI-keras_data/{species}/flanking_{flanking_size}/SpliceAI_{species}_train_{flanking_size}_{rs_idx}_rs{rs}/{rs_idx}/"
    elif experiment == "MANE_cleaned_test_set":
        res_root = f"/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/test_outdir/clean_test_dataset/{species}/flanking_{flanking_size}/SpliceAI_{species}_train_{flanking_size}_{rs_idx}_rs{rs}/{rs_idx}/"
    elif experiment == "MANE_cleaned_test_set_SpliceAI-keras_model":
        res_root = f"/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/test_outdir/clean_test_dataset_SpliceAI-keras_model/{species}/flanking_{flanking_size}/SpliceAI_{species}_train_{flanking_size}_{rs_idx}_rs{rs}/{rs_idx}/"

    if target == "SpliceAI-Keras":
        log_output_base = f"{res_root}TEST_LOG_SpliceAI-Keras/"
    elif target == "OpenSpliceAI":
        log_output_base = f"{res_root}TEST_LOG/"
    log_output_test_base = f"{log_output_base}TEST/"
    if not os.path.exists(log_output_test_base):
        sys.exit(f"Path does not exist: {log_output_test_base}")
    return log_output_test_base


def classwise_accuracy(true_classes, predicted_classes, num_classes):
    class_accuracies = []
    for i in range(num_classes):
        true_positives = np.sum((predicted_classes == i) & (true_classes == i))
        total_class_samples = np.sum(true_classes == i)
        if total_class_samples > 0:
            accuracy = true_positives / total_class_samples
        else:
            accuracy = 0.0  # Or set to an appropriate value for classes with no samples
        class_accuracies.append(accuracy)
    return class_accuracies


def collect_metrics(output_dir, sequence_length, random_seeds, species, experiment):
    """Collect metric values for each seed."""
    metrics_across_spliceai_keras = {
        'donor_topk': [],
        'donor_auprc': [],
        'donor_auroc': [],
        'donor_accuracy': [],
        'donor_precision': [],
        'donor_recall': [],
        'donor_f1': [],
        'acceptor_topk': [],
        'acceptor_auprc': [],
        'acceptor_auroc': [],
        'acceptor_accuracy': [],
        'acceptor_precision': [],
        'acceptor_recall': [],
        'acceptor_f1': []
    }
    metrics_across_spliceai_pytorch = {
        'donor_topk': [],
        'donor_auprc': [],
        'donor_auroc': [],
        'donor_accuracy': [],
        'donor_precision': [],
        'donor_recall': [],
        'donor_f1': [],
        'acceptor_topk': [],
        'acceptor_auprc': [],
        'acceptor_auroc': [],
        'acceptor_accuracy': [],
        'acceptor_precision': [],
        'acceptor_recall': [],
        'acceptor_f1': []
    }
    # Gather metrics for SpliceAI-Keras
    for flanking_size in [80, 400, 2000, 10000]:
        for rs_idx, rs in enumerate(random_seeds):
            log_output_test_base = initialize_paths(output_dir, flanking_size, sequence_length, rs, rs_idx, species, experiment=experiment, target="SpliceAI-Keras")
            metrics_for_spliceai_keras = {
                'donor_topk': f'{log_output_test_base}/donor_topk.txt',
                'donor_auprc': f'{log_output_test_base}/donor_auprc.txt',
                'donor_auroc': f'{log_output_test_base}/donor_auroc.txt',
                'donor_accuracy': f'{log_output_test_base}/donor_accuracy.txt',
                'donor_precision': f'{log_output_test_base}/donor_precision.txt',
                'donor_recall': f'{log_output_test_base}/donor_recall.txt',
                'donor_f1': f'{log_output_test_base}/donor_f1.txt',

                'acceptor_topk': f'{log_output_test_base}/acceptor_topk.txt',
                'acceptor_auprc': f'{log_output_test_base}/acceptor_auprc.txt',
                'acceptor_auroc': f'{log_output_test_base}/acceptor_auroc.txt',
                'acceptor_accuracy': f'{log_output_test_base}/acceptor_accuracy.txt',
                'acceptor_precision': f'{log_output_test_base}/acceptor_precision.txt',
                'acceptor_recall': f'{log_output_test_base}/acceptor_recall.txt',
                'acceptor_f1': f'{log_output_test_base}/acceptor_f1.txt',
            }
            print("metrics_for_spliceai_keras: ", metrics_for_spliceai_keras)
            for metric, filepath in metrics_for_spliceai_keras.items():
                try:
                    with open(filepath, 'r') as f:
                        print("filepath: ", filepath)
                        value = float(f.read().strip().split('\n')[-1])
                        print(f"Value for {metric} at seed {rs}: {value}. ({flanking_size})")
                        metrics_across_spliceai_keras[metric].append((rs, value, flanking_size))
                except FileNotFoundError:
                    print(f"File not found: {filepath}")


    # Gather metrics for SpliceAI-Pytorch 
    for flanking_size in [80, 400, 2000, 10000]:
        for rs_idx, rs in enumerate(random_seeds):
            log_output_test_base = initialize_paths(output_dir, flanking_size, sequence_length, rs, rs_idx, species, experiment=experiment, target="OpenSpliceAI")
            metrics_for_spliceai_pytorch = {
                'donor_topk': f'{log_output_test_base}/donor_topk.txt',
                'donor_auprc': f'{log_output_test_base}/donor_auprc.txt',
                'donor_auroc': f'{log_output_test_base}/donor_auroc.txt',
                'donor_accuracy': f'{log_output_test_base}/donor_accuracy.txt',
                'donor_precision': f'{log_output_test_base}/donor_precision.txt',
                'donor_recall': f'{log_output_test_base}/donor_recall.txt',
                'donor_f1': f'{log_output_test_base}/donor_f1.txt',

                'acceptor_topk': f'{log_output_test_base}/acceptor_topk.txt',
                'acceptor_auprc': f'{log_output_test_base}/acceptor_auprc.txt',
                'acceptor_auroc': f'{log_output_test_base}/acceptor_auroc.txt',
                'acceptor_accuracy': f'{log_output_test_base}/acceptor_accuracy.txt',
                'acceptor_precision': f'{log_output_test_base}/acceptor_precision.txt',
                'acceptor_recall': f'{log_output_test_base}/acceptor_recall.txt',
                'acceptor_f1': f'{log_output_test_base}/acceptor_f1.txt',
            }
            print("metrics_for_spliceai_pytorch: ", metrics_for_spliceai_pytorch)
            for metric, filepath in metrics_for_spliceai_pytorch.items():
                try:
                    with open(filepath, 'r') as f:
                        print("filepath: ", filepath)
                        value = float(f.read().strip().split('\n')[-1])
                        print(f"Value for {metric} at seed {rs}: {value}. ({flanking_size})")
                        metrics_across_spliceai_pytorch[metric].append((rs, value, flanking_size))
                except FileNotFoundError:
                    print(f"File not found: {filepath}")
    return metrics_across_spliceai_keras, metrics_across_spliceai_pytorch

        
def plot_metrics_with_error_bars(metrics_across_spliceai_keras, metrics_across_spliceai_pytorch, flanking_sizes, species, experiment):
    key_mappings = {
        'donor_topk': 'Donor Top-K',
        'donor_auprc': 'Donor AUPRC',
        'donor_accuracy': 'Donor Accuracy',
        'acceptor_topk': 'Acceptor Top-K',
        'acceptor_auprc': 'Acceptor AUPRC',
        'acceptor_accuracy': 'Acceptor Accuracy',
        'donor_precision': 'Donor Precision',
        'donor_recall': 'Donor Recall',
        'donor_f1': 'Donor F1',
        'acceptor_precision': 'Acceptor Precision',
        'acceptor_recall': 'Acceptor Recall',
        'acceptor_f1': 'Acceptor F1'
    }
    metrics_keys = list(key_mappings.keys())
    n_metrics = len(metrics_keys) // 4
    fig, axs = plt.subplots(4, n_metrics, figsize=(15,15), sharey=True)     
    fig.suptitle(f"Splice site prediction metrics for {species}", fontsize=24)
    # After creating subplots, adjust layout manually
    plt.tight_layout(pad=3.0, h_pad=5.0)  # h_pad is the padding (height) between rows of subplots
    plt.subplots_adjust(hspace=0.5)  # Adjust the height of the space between subplots
    for i, key in enumerate(metrics_keys):
        # Convert linear index to 2D index
        row, col = divmod(i, n_metrics)
        ax = axs[row, col]
        values_keras = []
        std_dev_values_keras = []
        values_pytorch = []
        std_dev_values_pytorch = []
        for flanking_size in flanking_sizes:
            keras_samples = [value for idx, value, fs in metrics_across_spliceai_keras[key] if fs == flanking_size]
            pytorch_samples = [value for idx, value, fs in metrics_across_spliceai_pytorch[key] if fs == flanking_size]
            values_keras.append(np.mean(keras_samples) if keras_samples else np.nan)
            std_dev_values_keras.append(np.std(keras_samples) if keras_samples else np.nan)
            values_pytorch.append(np.mean(pytorch_samples) if pytorch_samples else np.nan)
            std_dev_values_pytorch.append(np.std(pytorch_samples) if pytorch_samples else np.nan)

        # Setting x-ticks to be categorical
        x_ticks = np.arange(len(flanking_sizes))
        print("values_pytorch: ", values_pytorch)
        print("std_dev_values_pytorch: ", std_dev_values_pytorch)
        
        # Plotting
        ax.errorbar(x_ticks, values_keras, yerr=std_dev_values_keras, fmt='-o', capsize=5, label='SpliceAI-Keras(Human)')#, color='blue')
        ax.errorbar(x_ticks, values_pytorch, yerr=std_dev_values_pytorch, fmt='-X', capsize=5, label='SpliceAI-Pytorch(Human)')#, color='green')        
        ax.set_ylim(0, 1)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(flanking_sizes)
        ax.set_xlabel('Flanking Size')
        ax.set_ylabel(key_mappings[key])
        ax.set_title(f"{key_mappings[key]}", fontweight='bold')
        ax.grid(True)
        ax.legend()
    plt.savefig(f"viz_{experiment}/combined_metrics_{species}.png", dpi=300)

def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', '-p', type=str)
    parser.add_argument('--random-seeds', '-r', type=str)
    parser.add_argument('--project-name', '-s', type=str)
    parser.add_argument('--species', '-sp', type=str)
    parser.add_argument('--experiment', '-e', type=str, default="my_split_paralog_removal")
    args = parser.parse_args()
    print("args: ", args, file=sys.stderr)
    print("Visualizing SpliceAI-toolkit results")
    os.makedirs(f"viz_{args.experiment}", exist_ok=True)

    output_dir = args.output_dir
    sequence_length = 5000
    random_seeds = args.random_seeds
    random_seeds = [22]#, 22, 40]

    metrics_across_spliceai_keras, metrics_across_spliceai_pytorch = collect_metrics(output_dir, sequence_length, random_seeds, args.species, args.experiment)

    print("metrics_across_spliceai_keras: ", metrics_across_spliceai_keras)
    print("metrics_across_spliceai_pytorch: ", metrics_across_spliceai_pytorch)

    flanking_sizes = [80, 400, 2000, 10000]
    plot_metrics_with_error_bars(metrics_across_spliceai_keras, metrics_across_spliceai_pytorch, flanking_sizes, args.species, args.experiment)

    # plot_combined_metrics(metrics_across_spliceai_keras, metrics_across_spliceai_pytorch, flanking_sizes, args.species)


if __name__ == "__main__":
    predict()