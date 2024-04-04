import argparse
import os, sys
import numpy as np
import platform
import time
import matplotlib.pyplot as plt
import sys

RANDOM_SEED = 42

def initialize_paths(output_dir, flanking_size, sequence_length, idx, species, target):
    """Initialize project directories and create them if they don't exist."""
    ####################################
    # Modify the model verson here!!
    ####################################
    if target == "spliceai_keras":
        MODEL_VERSION = f"spliceai{idx}_{species}_{sequence_length}_{flanking_size}"
    elif target == "spliceai_pytorch":
        MODEL_VERSION = f"spliceai_{species}_rs{idx}_{sequence_length}_{flanking_size}"
        # MODEL_VERSION = f"RefSeq_noncoding_fine-tune_unfreeze_last_residual_{sequence_length}_{flanking_size}"
        # MODEL_VERSION = f"RefSeq_noncoding_rs22_{sequence_length}_{flanking_size}"
    ####################################
    # Modify the model verson here!!
    ####################################
    res_root = "/home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/results/model_predict_outdir"
    model_train_outdir = f"{res_root}/{MODEL_VERSION}/"
    model_output_base = f"{model_train_outdir}models/"
    log_output_base = f"{model_train_outdir}LOG/"
    log_output_test_base = f"{log_output_base}TEST/"
    for path in [model_output_base, log_output_test_base]:
        if not os.path.exists(path):
            sys.exit(f"Path does not exist: {path}")
    return model_output_base, log_output_test_base


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


def metrics(batch_ypred, batch_ylabel, metric_files):
    # Assuming batch_ylabel and batch_ypred are your input tensors
    # Convert softmax probabilities to predicted classes
    _, predicted_classes = torch.max(batch_ypred, 1)  # Ensure this matches your data shape correctly
    true_classes = torch.argmax(batch_ylabel, dim=1)  # Adjust the axis if necessary
    # Convert tensors to numpy for compatibility with scikit-learn
    true_classes = true_classes.numpy()
    predicted_classes = predicted_classes.numpy()
    # Flatten arrays if they're 2D (for multi-class, not multi-label)
    true_classes_flat = true_classes.flatten()
    predicted_classes_flat = predicted_classes.flatten()
    # Now, calculate the metrics without iterating over each class
    accuracy = accuracy_score(true_classes_flat, predicted_classes_flat)
    precision, recall, f1, _ = precision_recall_fscore_support(true_classes_flat, predicted_classes_flat, average=None)
    class_accuracies = classwise_accuracy(true_classes, predicted_classes, 3)
    # Print overall accuracy (not class-wise)
    overall_accuracy = np.mean(class_accuracies)
    print(f"Overall Accuracy: {overall_accuracy}")
    for k, v in metric_files.items():
        with open(v, 'a') as f:
            if k == "accuracy":
                f.write(f"{overall_accuracy}\n")
    # Iterate over each class to print/save the metrics
    ss_types = ["Non-splice", "acceptor", "donor"]
    for i, (acc, prec, rec, f1_score) in enumerate(zip(class_accuracies, precision, recall, f1)):
        print(f"Class {ss_types[i]}\t: Accuracy={acc}, Precision={prec}, Recall={rec}, F1={f1_score}")
        for k, v in metric_files.items():
            with open(v, 'a') as f:
                if k == f"{ss_types[i]}_precision":
                    f.write(f"{prec}\n")
                elif k == f"{ss_types[i]}_recall":
                    f.write(f"{rec}\n")
                elif k == f"{ss_types[i]}_f1":
                    f.write(f"{f1_score}\n")
                elif k == f"{ss_types[i]}_accuracy":
                    # Append class-wise accuracy under the general accuracy file
                    f.write(f"{acc}\n")
    

def collect_metrics(output_dir, sequence_length, random_seeds, species):
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
    # Plot SpliceAI-Keras
    for flanking_size in [80, 400, 2000]:#, 10000]:
        for idx in range(1,6):
            print(f"idx: {idx}")
            _, log_output_test_base = initialize_paths(output_dir, flanking_size, sequence_length, idx, species, target="spliceai_keras")
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
                        value = float(f.read().strip())
                        print(f"Value for {metric} at seed {idx}: {value}. ({flanking_size})")
                        metrics_across_spliceai_keras[metric].append((idx, value, flanking_size))
                except FileNotFoundError:
                    print(f"File not found: {filepath}")

    # Plot SpliceAI-Pytorch 
    for flanking_size in [80, 400, 2000, 10000]:
        for idx, rs in enumerate(random_seeds):
            _, log_output_test_base = initialize_paths(output_dir, flanking_size, sequence_length, rs, species, target="spliceai_pytorch")
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
                        value = float(f.read().strip().split('\n')[0])
                        print(f"Value for {metric} at seed {rs}: {value}. ({flanking_size})")
                        metrics_across_spliceai_pytorch[metric].append((rs, value, flanking_size))
                except FileNotFoundError:
                    print(f"File not found: {filepath}")
    # print("metrics_across_spliceai_keras: ", metrics_across_spliceai_keras)
    # print("metrics_across_spliceai_pytorch: ", metrics_across_spliceai_pytorch)
    return metrics_across_spliceai_keras, metrics_across_spliceai_pytorch

        
def plot_metrics_with_error_bars(metrics_across_spliceai_keras, metrics_across_spliceai_pytorch, flanking_sizes, species):
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
    # print("metrics_keys: ", metrics_keys)
    n_metrics = len(metrics_keys) // 4
    fig, axs = plt.subplots(4, n_metrics, figsize=(15,15), sharey=True)     
    # fig, axs = plt.subplots(2, n_metrics, figsize=(10,8), sharey=True)     
    # if n_metrics == 1:  # If there's only one metric, axs will not be an array
    #     axs = [axs]
    fig.suptitle(f"Splice site prediction metrics for {species}", fontsize=24)
    # After creating subplots, adjust layout manually
    plt.tight_layout(pad=3.0, h_pad=5.0)  # h_pad is the padding (height) between rows of subplots
    plt.subplots_adjust(hspace=0.5)  # Adjust the height of the space between subplots
    for i, key in enumerate(metrics_keys):
        # Convert linear index to 2D index
        row, col = divmod(i, n_metrics)
        print(f"key: {key};  Row: {row}, Col: {col}")
        ax = axs[row, col]
        mean_values_keras = []
        std_dev_values_keras = []
        values_pytorch = []
        std_dev_values_pytorch = []
        for flanking_size in flanking_sizes:
            keras_samples = [value for idx, value, fs in metrics_across_spliceai_keras[key] if fs == flanking_size]
            pytorch_samples = [value for idx, value, fs in metrics_across_spliceai_pytorch[key] if fs == flanking_size]
            print("keras_samples: ", keras_samples)
            print("pytorch_samples: ", pytorch_samples)
            mean_values_keras.append(np.mean(keras_samples) if keras_samples else np.nan)
            std_dev_values_keras.append(np.std(keras_samples) if keras_samples else np.nan)
            values_pytorch.append(np.mean(pytorch_samples) if pytorch_samples else np.nan)
            std_dev_values_pytorch.append(np.std(pytorch_samples) if keras_samples else np.nan)

        # Setting x-ticks to be categorical
        x_ticks = np.arange(len(flanking_sizes))
        
        # Plotting
        ax.errorbar(x_ticks, mean_values_keras, yerr=std_dev_values_pytorch, fmt='-o', capsize=5, label='SpliceAI-Keras(Human)')#, color='blue')

        ax.errorbar(x_ticks, values_pytorch, yerr=std_dev_values_keras, fmt='-X', capsize=5, label='SpliceAI-Pytorch(Human)')#, color='green')        
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(flanking_sizes)
        ax.set_xlabel('Flanking Size')
        ax.set_ylabel(key_mappings[key])
        ax.set_title(f"{key_mappings[key]}", fontweight='bold')
        ax.grid(True)
        ax.legend()
    plt.savefig(f"vis/combined_metrics_{species}.png", dpi=300)

def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', '-p', type=str)
    parser.add_argument('--random-seeds', '-r', type=str)
    parser.add_argument('--project-name', '-s', type=str)
    parser.add_argument('--species', '-sp', type=str)
    args = parser.parse_args()
    print("args: ", args, file=sys.stderr)
    print("Visualizing SpliceAI-toolkit results")

    output_dir = args.output_dir
    sequence_length = 5000
    random_seeds = args.random_seeds
    # random_seeds = [15, 22, 30, 40]
    # random_seeds = [11, 12, 22, 40]
    random_seeds = [1, 2]#, 22, 40]

    metrics_across_spliceai_keras, metrics_across_spliceai_pytorch = collect_metrics(output_dir, sequence_length, random_seeds, args.species)

    flanking_sizes = [80, 400, 2000, 10000]
    plot_metrics_with_error_bars(metrics_across_spliceai_keras, metrics_across_spliceai_pytorch, flanking_sizes, args.species)
    # plot_combined_metrics(metrics_across_spliceai_keras, metrics_across_spliceai_pytorch, flanking_sizes, args.species)


if __name__ == "__main__":
    predict()