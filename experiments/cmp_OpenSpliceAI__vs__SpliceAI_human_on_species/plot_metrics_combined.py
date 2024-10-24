# import argparse
# import os, sys
# import numpy as np
# import platform
# import time
# import matplotlib.pyplot as plt
# import sys

# RANDOM_SEED = 42

# def initialize_paths(output_dir, flanking_size, sequence_length, rs, rs_idx, species, experiment, target):
#     """Initialize project directories and create them if they don't exist."""
#     # Experiment paths
#     res_root = f"/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/test_outdir/test_human_model_on_species/{species}/flanking_{flanking_size}/SpliceAI_{species}_train_{flanking_size}_{rs_idx}_rs{rs}/{rs_idx}/"

#     if experiment == "arabidopsis":
#         pass
#     elif experiment == "honeybee":
#         pass
#     elif experiment == "mouse":
#         pass
#     elif experiment == "zebrafish":
#         pass

#     log_output_base = f"{res_root}"
#     if target == "SpliceAI-Keras":
#         log_output_base = f"{res_root}TEST_LOG_SPLICEAI_KERAS_{experiment}/"
#     elif target == "OpenSpliceAI":
#         log_output_base = f"{res_root}TEST_LOG_{experiment}/"
#     log_output_test_base = f"{log_output_base}TEST/"
#     if not os.path.exists(log_output_test_base):
#         sys.exit(f"Path does not exist: {log_output_test_base}")
#     return log_output_test_base


# def classwise_accuracy(true_classes, predicted_classes, num_classes):
#     class_accuracies = []
#     for i in range(num_classes):
#         true_positives = np.sum((predicted_classes == i) & (true_classes == i))
#         total_class_samples = np.sum(true_classes == i)
#         if total_class_samples > 0:
#             accuracy = true_positives / total_class_samples
#         else:
#             accuracy = 0.0  # Or set to an appropriate value for classes with no samples
#         class_accuracies.append(accuracy)
#     return class_accuracies


# def collect_metrics(output_dir, sequence_length, random_seeds, species, experiment):
#     """Collect metric values for each seed."""
#     metrics_across_spliceai_keras = {
#         'donor_topk': [],
#         'donor_auprc': [],
#         'donor_auroc': [],
#         'donor_accuracy': [],
#         'donor_precision': [],
#         'donor_recall': [],
#         'donor_f1': [],
#         'acceptor_topk': [],
#         'acceptor_auprc': [],
#         'acceptor_auroc': [],
#         'acceptor_accuracy': [],
#         'acceptor_precision': [],
#         'acceptor_recall': [],
#         'acceptor_f1': []
#     }
#     metrics_across_spliceai_pytorch = {
#         'donor_topk': [],
#         'donor_auprc': [],
#         'donor_auroc': [],
#         'donor_accuracy': [],
#         'donor_precision': [],
#         'donor_recall': [],
#         'donor_f1': [],
#         'acceptor_topk': [],
#         'acceptor_auprc': [],
#         'acceptor_auroc': [],
#         'acceptor_accuracy': [],
#         'acceptor_precision': [],
#         'acceptor_recall': [],
#         'acceptor_f1': []
#     }
#     # Gather metrics for SpliceAI-Keras
#     for flanking_size in [80, 400, 2000, 10000]:
#         for rs_idx, rs in enumerate(random_seeds):
#             log_output_test_base = initialize_paths(output_dir, flanking_size, sequence_length, rs, rs_idx, species, experiment=experiment, target="SpliceAI-Keras")
#             metrics_for_spliceai_keras = {
#                 'donor_topk': f'{log_output_test_base}/donor_topk.txt',
#                 'donor_auprc': f'{log_output_test_base}/donor_auprc.txt',
#                 'donor_auroc': f'{log_output_test_base}/donor_auroc.txt',
#                 'donor_accuracy': f'{log_output_test_base}/donor_accuracy.txt',
#                 'donor_precision': f'{log_output_test_base}/donor_precision.txt',
#                 'donor_recall': f'{log_output_test_base}/donor_recall.txt',
#                 'donor_f1': f'{log_output_test_base}/donor_f1.txt',

#                 'acceptor_topk': f'{log_output_test_base}/acceptor_topk.txt',
#                 'acceptor_auprc': f'{log_output_test_base}/acceptor_auprc.txt',
#                 'acceptor_auroc': f'{log_output_test_base}/acceptor_auroc.txt',
#                 'acceptor_accuracy': f'{log_output_test_base}/acceptor_accuracy.txt',
#                 'acceptor_precision': f'{log_output_test_base}/acceptor_precision.txt',
#                 'acceptor_recall': f'{log_output_test_base}/acceptor_recall.txt',
#                 'acceptor_f1': f'{log_output_test_base}/acceptor_f1.txt',
#             }
#             print("metrics_for_spliceai_keras: ", metrics_for_spliceai_keras)
#             for metric, filepath in metrics_for_spliceai_keras.items():
#                 try:
#                     with open(filepath, 'r') as f:
#                         print("filepath: ", filepath)
#                         value = float(f.read().strip().split('\n')[-1])
#                         print(f"Value for {metric} at seed {rs}: {value}. ({flanking_size})")
#                         metrics_across_spliceai_keras[metric].append((rs, value, flanking_size))
#                 except FileNotFoundError:
#                     print(f"File not found: {filepath}")


#     # Gather metrics for SpliceAI-Pytorch 
#     for flanking_size in [80, 400, 2000, 10000]:
#         for rs_idx, rs in enumerate(random_seeds):
#             log_output_test_base = initialize_paths(output_dir, flanking_size, sequence_length, rs, rs_idx, species, experiment=experiment, target="OpenSpliceAI")
#             metrics_for_spliceai_pytorch = {
#                 'donor_topk': f'{log_output_test_base}/donor_topk.txt',
#                 'donor_auprc': f'{log_output_test_base}/donor_auprc.txt',
#                 'donor_auroc': f'{log_output_test_base}/donor_auroc.txt',
#                 'donor_accuracy': f'{log_output_test_base}/donor_accuracy.txt',
#                 'donor_precision': f'{log_output_test_base}/donor_precision.txt',
#                 'donor_recall': f'{log_output_test_base}/donor_recall.txt',
#                 'donor_f1': f'{log_output_test_base}/donor_f1.txt',

#                 'acceptor_topk': f'{log_output_test_base}/acceptor_topk.txt',
#                 'acceptor_auprc': f'{log_output_test_base}/acceptor_auprc.txt',
#                 'acceptor_auroc': f'{log_output_test_base}/acceptor_auroc.txt',
#                 'acceptor_accuracy': f'{log_output_test_base}/acceptor_accuracy.txt',
#                 'acceptor_precision': f'{log_output_test_base}/acceptor_precision.txt',
#                 'acceptor_recall': f'{log_output_test_base}/acceptor_recall.txt',
#                 'acceptor_f1': f'{log_output_test_base}/acceptor_f1.txt',
#             }
#             print("metrics_for_spliceai_pytorch: ", metrics_for_spliceai_pytorch)
#             for metric, filepath in metrics_for_spliceai_pytorch.items():
#                 try:
#                     with open(filepath, 'r') as f:
#                         print("filepath: ", filepath)
#                         value = float(f.read().strip().split('\n')[-1])
#                         print(f"Value for {metric} at seed {rs}: {value}. ({flanking_size})")
#                         metrics_across_spliceai_pytorch[metric].append((rs, value, flanking_size))
#                 except FileNotFoundError:
#                     print(f"File not found: {filepath}")
#     return metrics_across_spliceai_keras, metrics_across_spliceai_pytorch


# def plot_metrics_with_error_bars(metrics_across_spliceai_keras, metrics_across_spliceai_pytorch, flanking_sizes, species, experiment):
#     key_mappings = {
#         'donor_topk': 'Donor Top-K',
#         'donor_auprc': 'Donor AUPRC',
#         'donor_accuracy': 'Donor Accuracy',
#         'acceptor_topk': 'Acceptor Top-K',
#         'acceptor_auprc': 'Acceptor AUPRC',
#         'acceptor_accuracy': 'Acceptor Accuracy',
#         'donor_precision': 'Donor Precision',
#         'donor_recall': 'Donor Recall',
#         'donor_f1': 'Donor F1',
#         'acceptor_precision': 'Acceptor Precision',
#         'acceptor_recall': 'Acceptor Recall',
#         'acceptor_f1': 'Acceptor F1'
#     }
#     metrics_keys = list(key_mappings.keys())
#     n_metrics = len(metrics_keys) // 4
#     fig, axs = plt.subplots(4, n_metrics, figsize=(15,15), sharey=True)     
#     fig.suptitle(f"Splice site prediction metrics for {species}", fontsize=24)
#     # After creating subplots, adjust layout manually
#     plt.tight_layout(pad=3.0, h_pad=5.0)  # h_pad is the padding (height) between rows of subplots
#     plt.subplots_adjust(hspace=0.5)  # Adjust the height of the space between subplots
#     for i, key in enumerate(metrics_keys):
#         # Convert linear index to 2D index
#         row, col = divmod(i, n_metrics)
#         ax = axs[row, col]
#         values_keras = []
#         std_dev_values_keras = []
#         values_pytorch = []
#         std_dev_values_pytorch = []
#         for flanking_size in flanking_sizes:
#             keras_samples = [value for idx, value, fs in metrics_across_spliceai_keras[key] if fs == flanking_size]
#             pytorch_samples = [value for idx, value, fs in metrics_across_spliceai_pytorch[key] if fs == flanking_size]
#             values_keras.append(np.mean(keras_samples) if keras_samples else np.nan)
#             std_dev_values_keras.append(np.std(keras_samples) if keras_samples else np.nan)
#             values_pytorch.append(np.mean(pytorch_samples) if pytorch_samples else np.nan)
#             std_dev_values_pytorch.append(np.std(pytorch_samples) if pytorch_samples else np.nan)

#         # Setting x-ticks to be categorical
#         x_ticks = np.arange(len(flanking_sizes))
#         print("values_pytorch: ", values_pytorch)
#         print("std_dev_values_pytorch: ", std_dev_values_pytorch)
        
#         # Plotting
#         ax.errorbar(x_ticks, values_keras, yerr=std_dev_values_keras, fmt='-o', capsize=5, label='SpliceAI-Keras(Human)')#, color='blue')
#         ax.errorbar(x_ticks, values_pytorch, yerr=std_dev_values_pytorch, fmt='-X', capsize=5, label='SpliceAI-Pytorch(Human)')#, color='green')        
#         ax.set_ylim(0, 1)
#         ax.set_xticks(x_ticks)
#         ax.set_xticklabels(flanking_sizes)
#         ax.set_xlabel('Flanking Size')
#         ax.set_ylabel(key_mappings[key])
#         ax.set_title(f"{key_mappings[key]}", fontweight='bold')
#         ax.grid(True)
#         ax.legend()
#     plt.savefig(f"viz_{experiment}/combined_metrics_{species}.png", dpi=300)


# def predict():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--output-dir', '-p', type=str)
#     parser.add_argument('--random-seeds', '-r', type=str)
#     parser.add_argument('--project-name', '-s', type=str)
#     parser.add_argument('--species', '-sp', type=str)
#     parser.add_argument('--experiment', '-e', type=str, default="my_split_paralog_removal")
#     args = parser.parse_args()
#     print("args: ", args, file=sys.stderr)
#     print("Visualizing SpliceAI-toolkit results")
#     os.makedirs(f"viz_{args.experiment}", exist_ok=True)

#     output_dir = args.output_dir
#     sequence_length = 5000
#     random_seeds = args.random_seeds
#     random_seeds = [22]#, 22, 40]

#     print("output_dir: ", output_dir)
#     print("sequence_length: ", sequence_length)
#     print("random_seeds: ", random_seeds)
    

#     metrics_across_spliceai_keras, metrics_across_spliceai_pytorch = collect_metrics(output_dir, sequence_length, random_seeds, args.species, args.experiment)

#     print("metrics_across_spliceai_keras: ", metrics_across_spliceai_keras)
#     print("metrics_across_spliceai_pytorch: ", metrics_across_spliceai_pytorch)

#     flanking_sizes = [80, 400, 2000, 10000]
#     plot_metrics_with_error_bars(metrics_across_spliceai_keras, metrics_across_spliceai_pytorch, flanking_sizes, args.species, args.experiment)

#     # plot_combined_metrics(metrics_across_spliceai_keras, metrics_across_spliceai_pytorch, flanking_sizes, args.species)


# if __name__ == "__main__":
#     predict()






import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math

def initialize_paths(flanking_size, rs, rs_idx, species, experiment, target):
    """Construct and return the test output path based on input parameters."""
    # Construct the base result directory
    res_root = os.path.join(
        "/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/test_outdir/",
        "test_human_model_on_species",
        species,
        f"flanking_{flanking_size}",
        f"SpliceAI_{species}_train_{flanking_size}_{rs_idx}_rs{rs}",
        str(rs_idx)
    )

    # Determine the log output base directory based on the target
    if target == "SpliceAI-Keras":
        log_output_base = os.path.join(res_root, f"TEST_LOG_SPLICEAI_KERAS_{experiment}")
    elif target == "OpenSpliceAI-MANE":
        log_output_base = os.path.join(res_root, f"TEST_LOG_{experiment}")
    elif target == "OpenSpliceAI-GENCODE":
        log_output_base = os.path.join(res_root, f"TEST_LOG_{experiment}_gencode_trained")
    else:
        log_output_base = res_root

    # Construct the final test output directory
    log_output_test_base = os.path.join(log_output_base, "TEST")

    # Check if the path exists
    if not os.path.exists(log_output_test_base):
        sys.exit(f"Path does not exist: {log_output_test_base}")
    return log_output_test_base


def collect_metrics_for_target(random_seeds, species, experiment, target, flanking_sizes):
    """Collect metrics for a given target."""
    metrics_across = {
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

    for flanking_size in flanking_sizes:
        for rs_idx, rs in enumerate(random_seeds):
            log_output_test_base = initialize_paths(
                flanking_size, rs, rs_idx, species, experiment, target
            )

            # Map metrics to their corresponding file paths
            metrics_files = {
                metric: os.path.join(log_output_test_base, f"{metric}.txt")
                for metric in metrics_across.keys()
            }

            print(f"Collecting metrics for {target} at flanking size {flanking_size}, seed {rs}")
            for metric, filepath in metrics_files.items():
                try:
                    with open(filepath, 'r') as f:
                        print(f"Reading file: {filepath}")
                        value = float(f.read().strip().split('\n')[-1])
                        print(f"Value for {metric} at seed {rs}: {value}. (Flanking size: {flanking_size})")
                        metrics_across[metric].append((rs, value, flanking_size))
                except FileNotFoundError:
                    print(f"File not found: {filepath}")
    return metrics_across


def collect_metrics(random_seeds, species, experiment, flanking_sizes, model_trained_on):
    """Collect metric values for each seed and target."""
    metrics_across_spliceai_keras = collect_metrics_for_target(
        random_seeds, species, experiment, target="SpliceAI-Keras", flanking_sizes=flanking_sizes
    )
    if model_trained_on == "MANE":
        metrics_across_spliceai_pytorch = collect_metrics_for_target(
            random_seeds, species, experiment, target="OpenSpliceAI-MANE", flanking_sizes=flanking_sizes
        )
    elif model_trained_on == "GENCODE":
        metrics_across_spliceai_pytorch = collect_metrics_for_target(
            random_seeds, species, experiment, target="OpenSpliceAI-GENCODE", flanking_sizes=flanking_sizes
        )
    return metrics_across_spliceai_keras, metrics_across_spliceai_pytorch


def plot_metrics_with_error_bars(output_dir, metrics_keras, metrics_pytorch, flanking_sizes, species, experiment):
    key_mappings = {
        'donor_topk': 'Donor Top-K',
        'donor_auprc': 'Donor AUPRC',
        # 'donor_auroc': 'Donor AUROC',
        'donor_accuracy': 'Donor Accuracy',
        'donor_precision': 'Donor Precision',
        'donor_recall': 'Donor Recall',
        'donor_f1': 'Donor F1',
        'acceptor_topk': 'Acceptor Top-K',
        'acceptor_auprc': 'Acceptor AUPRC',
        # 'acceptor_auroc': 'Acceptor AUROC',
        'acceptor_accuracy': 'Acceptor Accuracy',
        'acceptor_precision': 'Acceptor Precision',
        'acceptor_recall': 'Acceptor Recall',
        'acceptor_f1': 'Acceptor F1'
    }

    metrics_keys = list(key_mappings.keys())
    n_metrics = len(metrics_keys)
    n_cols = 3
    n_rows = math.ceil(n_metrics / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), sharey=False)
    fig.suptitle(f"Splice site prediction metrics for {species}", fontsize=24)
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(hspace=0.5)
    axs = axs.flatten()

    for i, key in enumerate(metrics_keys):
        ax = axs[i]
        values_keras = []
        std_dev_keras = []
        values_pytorch = []
        std_dev_pytorch = []

        for flanking_size in flanking_sizes:
            keras_samples = [
                value for idx, value, fs in metrics_keras[key] if fs == flanking_size
            ]
            pytorch_samples = [
                value for idx, value, fs in metrics_pytorch[key] if fs == flanking_size
            ]

            values_keras.append(np.mean(keras_samples) if keras_samples else np.nan)
            std_dev_keras.append(np.std(keras_samples) if keras_samples else np.nan)
            values_pytorch.append(np.mean(pytorch_samples) if pytorch_samples else np.nan)
            std_dev_pytorch.append(np.std(pytorch_samples) if pytorch_samples else np.nan)

        x_ticks = np.arange(len(flanking_sizes))

        ax.errorbar(
            x_ticks, values_keras, yerr=std_dev_keras, fmt='-o', capsize=5,
            label='SpliceAI-Keras(Human)'
        )
        ax.errorbar(
            x_ticks, values_pytorch, yerr=std_dev_pytorch, fmt='-X', capsize=5,
            label='SpliceAI-Pytorch(Human)'
        )
        ax.set_ylim(0, 1)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(flanking_sizes)
        ax.set_xlabel('Flanking Size')
        ax.set_ylabel(key_mappings[key])
        ax.set_title(key_mappings[key], fontweight='bold')
        ax.grid(True)
        ax.legend()

    # Remove any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.savefig(f"{output_dir}/combined_metrics_{species}.png", dpi=300)


def main():
    parser = argparse.ArgumentParser(description="Visualize SpliceAI-toolkit results.")
    parser.add_argument('--output-dir', '-p', type=str, required=True, help='Base output directory.')
    parser.add_argument('--random-seeds', '-r', type=str, default='22,40', help='Comma-separated list of random seeds.')
    parser.add_argument('--species', '-sp', type=str, required=True, help='Species name.')
    parser.add_argument('--experiment', '-e', type=str, default="my_split_paralog_removal", help='Experiment name.')
    parser.add_argument('--model-trained-on', '-m', type=str, default="MANE", help='Model trained on species.')
    args = parser.parse_args()

    print("args:", args, file=sys.stderr)
    print("Visualizing SpliceAI-toolkit results")

    os.makedirs(args.output_dir, exist_ok=True)

    random_seeds = [int(seed.strip()) for seed in args.random_seeds.split(',')]
    flanking_sizes = [80, 400, 2000, 10000]

    print("Random seeds:", random_seeds)

    metrics_keras, metrics_pytorch = collect_metrics(
        random_seeds, args.species, args.experiment, flanking_sizes,
        args.model_trained_on
    )

    print("Metrics across SpliceAI-Keras:", metrics_keras)
    print("Metrics across SpliceAI-PyTorch:", metrics_pytorch)

    plot_metrics_with_error_bars(
        args.output_dir, metrics_keras, metrics_pytorch, flanking_sizes, args.species, args.experiment
    )

if __name__ == "__main__":
    main()
