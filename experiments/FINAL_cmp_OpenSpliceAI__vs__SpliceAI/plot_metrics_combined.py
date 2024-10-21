import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math

def initialize_paths(flanking_size, rs, rs_idx, species, experiment, target):
    """Construct and return the test output path based on input parameters."""   
    # Construct the base result directory
    if target == "SpliceAI-Keras":
        res_root = os.path.join(
            "/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/test_outdir/",
            "FINAL",
            species,
            f"flanking_{flanking_size}",
            f"SpliceAI_{species}_train_{flanking_size}_{rs_idx}_rs{rs}",
            str(rs_idx)
        )
        log_output_base = os.path.join(res_root, f"TEST_LOG_SPLICEAI_KERAS")
    elif target == "OpenSpliceAI-MANE":
        res_root = os.path.join(
            "/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/test_outdir/",
            "FINAL",
            species,
            f"flanking_{flanking_size}",
            f"SpliceAI_{species}_train_{flanking_size}_{rs_idx}_rs{rs}",
            str(rs_idx)
        )
        log_output_base = os.path.join(res_root, f"TEST_LOG_OPENSPLICEAI")

    elif target == "OpenSpliceAI-Species":
        res_root = os.path.join(
            "/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/test_outdir/",
            "FINAL",
            species,
            f"flanking_{flanking_size}",
            f"SpliceAI_{species}_train_{flanking_size}_{rs_idx}_rs{rs}",
            str(rs_idx)
        )
        log_output_base = os.path.join(res_root, f"TEST_LOG_OPENSPLICEAI_{species}")
    else:
        log_output_base = res_root

    # Construct the final test output directory
    log_output_test_base = os.path.join(log_output_base, "TEST")
    print(f'{target} log_output_test_base: ', log_output_test_base)
    # Check if the path exists
    if not os.path.exists(log_output_test_base):
        # sys.exit(f"Path does not exist: {log_output_test_base}")
        print(f"Path does not exist: {log_output_test_base}")
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


def collect_metrics(random_seeds, species, experiment, flanking_sizes, openspliceai_model_type):
    """Collect metric values for each seed and target."""
    metrics_across_spliceai_keras = collect_metrics_for_target(
        random_seeds, species, experiment, target="SpliceAI-Keras", flanking_sizes=flanking_sizes
    )

    # model-type
    if openspliceai_model_type == "MANE":
        metrics_across_spliceai_pytorch = collect_metrics_for_target(
            random_seeds, species, experiment, target="OpenSpliceAI-MANE", flanking_sizes=flanking_sizes
        )        
    elif openspliceai_model_type == "species":
        metrics_across_spliceai_pytorch = collect_metrics_for_target(
            random_seeds, species, experiment, target="OpenSpliceAI-Species", flanking_sizes=flanking_sizes
        )
    return metrics_across_spliceai_keras, metrics_across_spliceai_pytorch


def plot_metrics_with_error_bars(output_dir, metrics_keras, metrics_pytorch, flanking_sizes, species, experiment):
    key_mappings = {
        'donor_topk': 'Donor Top-K',
        'donor_auprc': 'Donor AUPRC',
        # 'donor_auroc': 'Donor AUROC',
        # 'donor_accuracy': 'Donor Accuracy',
        'donor_precision': 'Donor Precision',
        'donor_recall': 'Donor Recall',
        'donor_f1': 'Donor F1',
        'acceptor_topk': 'Acceptor Top-K',
        'acceptor_auprc': 'Acceptor AUPRC',
        # 'acceptor_auroc': 'Acceptor AUROC',
        # 'acceptor_accuracy': 'Acceptor Accuracy',
        'acceptor_precision': 'Acceptor Precision',
        'acceptor_recall': 'Acceptor Recall',
        'acceptor_f1': 'Acceptor F1'
    }


    if species == "MANE":
        spliceai_keras_label = f'SpliceAI-Keras (Human-MANE)'
        openspliceai_label = f'OpenSpliceAI (Human-MANE)'
        title = f"Splice site prediction metrics for Human-MANE"
    elif species == "honeybee":
        spliceai_keras_label = f'SpliceAI-Keras (Honeybee)'
        openspliceai_label = f'OpenSpliceAI (Honeybee)'
        title = f"Splice site prediction metrics for Honeybee"
    elif species == "arabidopsis":
        spliceai_keras_label = f'SpliceAI-Keras (Thale Cress)'
        openspliceai_label = f'OpenSpliceAI (Thale Cress)'
        title = f"Splice site prediction metrics for Thale Cress"
    elif species == "zebrafish":
        spliceai_keras_label = f'SpliceAI-Keras (Zebrafish)'
        openspliceai_label = f'OpenSpliceAI (Zebrafish)'
        title = f"Splice site prediction metrics for Zebrafish"
    elif species == "mouse":
        spliceai_keras_label = f'SpliceAI-Keras (Mouse)'
        openspliceai_label = f'OpenSpliceAI (Mouse)'
        title = f"Splice site prediction metrics for Mouse"

    metrics_keys = list(key_mappings.keys())
    n_metrics = len(metrics_keys)
    n_cols = 5
    n_rows = math.ceil(n_metrics / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows), sharey=False)
    fig.suptitle(title, fontsize=24)
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
            label=spliceai_keras_label
        )
        ax.errorbar(
            x_ticks, values_pytorch, yerr=std_dev_pytorch, fmt='-X', capsize=5,
            label=openspliceai_label
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

    plt.savefig(f"{output_dir}/combined_metrics_{species}_selected.png", dpi=300)


def main():
    parser = argparse.ArgumentParser(description="Visualize SpliceAI-toolkit results.")
    parser.add_argument('--output-dir', '-p', type=str, required=True, help='Base output directory.')
    parser.add_argument('--species', '-sp', type=str, required=True, help='Species name.')
    parser.add_argument('--experiment', '-e', type=str, default="", help='Experiment name.')
    parser.add_argument('--openspliceai-model-type', '-m', type=str, default="MANE", choices=["MANE", "species"], help='Model type.')
    # parser.add_argument('--selected', '-s', action='store_true', default=False, help='Selected metrics')
                        
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random_seeds = [10, 11, 12, 13, 14]
    flanking_sizes = [80, 400, 2000, 10000]

    print("args:", args, file=sys.stderr)
    print("Random seeds:", random_seeds)

    metrics_keras, metrics_pytorch = collect_metrics(
        random_seeds, args.species, args.experiment, flanking_sizes, args.openspliceai_model_type
    )

    print("Metrics across SpliceAI-Keras:", metrics_keras)
    print("Metrics across OpenSpliceAI:", metrics_pytorch)

    plot_metrics_with_error_bars(
        args.output_dir, metrics_keras, metrics_pytorch, flanking_sizes, 
        args.species, args.experiment
    )

if __name__ == "__main__":
    main()
