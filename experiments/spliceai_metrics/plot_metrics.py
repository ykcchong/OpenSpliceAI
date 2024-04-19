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
    #     os.makedirs(path, exist_ok=True)
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
    

def collect_metrics(output_dir, flanking_size, sequence_length, random_seeds, species):
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
    for idx in range(1,4):
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
                    print(f"Value for {metric} at seed {idx}: {value}")
                    metrics_across_spliceai_keras[metric].append((idx, value))
            except FileNotFoundError:
                print(f"File not found: {filepath}")

    # Plot SpliceAI-Pytorch 
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
                print(f"filepath: {filepath}")
                with open(filepath, 'r') as f:
                    value = float(f.read().strip())
                    print(f"Value for {metric} at seed {rs}: {value}")
                    metrics_across_spliceai_pytorch[metric].append((rs, value))
            except FileNotFoundError:
                print(f"File not found: {filepath}")
    return metrics_across_spliceai_keras, metrics_across_spliceai_pytorch


def plot_combined_metrics(metrics_across_spliceai_keras, metrics_across_spliceai_pytorch, flanking_size, species):
    """Plot collected metrics across different random seeds with the same x-axis value."""
    keys = ['donor_topk', 'donor_auprc', 'donor_accuracy', 'donor_precision', 'donor_recall', 'donor_f1', 
            'acceptor_topk', 'acceptor_auprc', 'acceptor_accuracy', 'acceptor_precision', 'acceptor_recall', 'acceptor_f1']
    n_metrics = len(keys)
    n_cols = n_metrics // 2 + (n_metrics % 2 > 0)  # Calculate the number of columns needed for two rows
    fig, axs = plt.subplots(2, n_cols, figsize=(15, 5), sharey='row')  # Adjust figsize as needed
    # fig.suptitle('Combined Metrics for SpliceAI-10k-Keras and SpliceAI-10k-Pytorch')

    # Settings for both sets of metrics
    x_positions = np.array([1, 2])  # Fixed x-axis values for all points
    jitter_width = 0.1  # Width for jitter to avoid overlap

    for i, metric in enumerate(keys):
        row, col = divmod(i, n_cols)
        ax = axs[row, col]
        
        # Keras Metrics
        if metric in metrics_across_spliceai_keras:
            values = metrics_across_spliceai_keras[metric]
            jittered_x_positions = x_positions[0] + np.random.uniform(-jitter_width, jitter_width, size=len(values))
            _, vals = zip(*values)
            ax.scatter(jittered_x_positions, vals, alpha=0.6, label='SpliceAI-10k-Keras')

        # PyTorch Metrics
        if metric in metrics_across_spliceai_pytorch:
            values = metrics_across_spliceai_pytorch[metric]
            jittered_x_positions = x_positions[1] + np.random.uniform(-jitter_width, jitter_width, size=len(values))
            _, vals = zip(*values)
            ax.scatter(jittered_x_positions, vals, alpha=0.6, label='SpliceAI-10k-Pytorch')

        ax.set_title(metric)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(['Keras', 'PyTorch'])
        ax.set_xlabel('Framework')
        ax.set_ylabel('Value')

    # Create a shared legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize='large', ncol = 2)

    plt.tight_layout()
    # Adjust the layout to make space for the legend
    fig.subplots_adjust(top=0.85)  # You might need to adjust this value based on your figure's dimensions
    print(f"Output figure: vis/metrics_{species}_{flanking_size}.png")
    plt.savefig(f"vis/metrics_{species}_{flanking_size}.png")




# def plot_metrics(test_metric_files, rs):
#     """Plot each metric in a separate subplot."""
#     n = len(test_metric_files)
#     # Create subplots arranged in a grid with a suitable number of rows
#     nrows = int(n**0.5) + 1
#     ncols = (n // nrows) + (n % nrows > 0)
    
#     fig, axs = plt.subplots(nrows, ncols, figsize=(15, 10)) # Adjust the size as needed
#     fig.suptitle('SpliceAI-toolkit Metric Visualizations')
    
#     for idx, (metric_name, file_path) in enumerate(test_metric_files.items()):
#         # Read the metric value from the file
#         try:
#             with open(file_path, 'r') as file:
#                 value = float(file.readline().strip())
#         except FileNotFoundError:
#             print(f"File not found: {file_path}")
#             value = None
        
#         # Plot the value
#         if value is not None:
#             ax = axs[idx // ncols, idx % ncols]
#             ax.scatter([1], [value])  # Plotting the value at x=1
#             ax.set_title(metric_name)
#             ax.set_xticks([1])  # Since we only have one x-value
#             ax.set_xticklabels(['Value'])
#             ax.grid(True)
        
#     # Adjust layout to make room for the title and ensure plots don't overlap
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)
#     plt.savefig(f"vis/metrics_rs{rs}.png")


def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', '-p', type=str)
    parser.add_argument('--flanking-size', '-f', type=int, default=80)
    parser.add_argument('--random-seeds', '-r', type=str)
    parser.add_argument('--project-name', '-s', type=str)
    parser.add_argument('--species', '-sp', type=str)
    args = parser.parse_args()
    print("args: ", args, file=sys.stderr)
    print("Visualizing SpliceAI-toolkit results")

    
    output_dir = args.output_dir
    sequence_length = 5000
    flanking_size = int(args.flanking_size)
    random_seeds = args.random_seeds
    random_seeds = [1, 2]
    # random_seeds = [15, 22, 30, 40]
    # random_seeds = [11, 12, 22, 40]

    metrics_across_spliceai_keras, metrics_across_spliceai_pytorch = collect_metrics(output_dir, flanking_size, sequence_length, random_seeds, args.species)
    plot_combined_metrics(metrics_across_spliceai_keras, metrics_across_spliceai_pytorch, flanking_size, args.species)

if __name__ == "__main__":
    predict()