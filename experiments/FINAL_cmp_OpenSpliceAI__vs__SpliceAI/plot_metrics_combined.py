import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import ttest_rel, wilcoxon

def cohens_d(x, y):
    diff = np.array(x) - np.array(y)
    return np.mean(diff) / np.std(diff, ddof=1)

def calculate_improvement(metrics, flanking_sizes):
    """Calculate the percentage improvement as flanking size increases."""
    improvements = {}
    for metric, values in metrics.items():
        improvements[metric] = []
        for i in range(1, len(flanking_sizes)):
            # Get values for the two flanking sizes being compared
            prev_values = [value for idx, value, fs in values if fs == flanking_sizes[i-1]]
            curr_values = [value for idx, value, fs in values if fs == flanking_sizes[i]]
            
            if prev_values and curr_values:
                prev_mean = np.mean(prev_values)
                curr_mean = np.mean(curr_values)
                # Calculate percentage improvement
                improvement = ((curr_mean - prev_mean) / prev_mean) * 100 if prev_mean != 0 else np.nan
                improvements[metric].append((flanking_sizes[i-1], flanking_sizes[i], improvement))
            else:
                improvements[metric].append((flanking_sizes[i-1], flanking_sizes[i], np.nan))
    
    return improvements

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
        # 'donor_auroc': [],
        'donor_accuracy': [],
        'donor_precision': [],
        'donor_recall': [],
        'donor_f1': [],
        'acceptor_topk': [],
        'acceptor_auprc': [],
        # 'acceptor_auroc': [],
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
        'donor_topk': 'Donor Top-1',
        # 'donor_auprc': 'Donor AUPRC',
        # 'donor_auroc': 'Donor AUROC',
        # 'donor_accuracy': 'Donor Accuracy',
        # 'donor_precision': 'Donor Precision',
        # 'donor_recall': 'Donor Recall',
        'donor_f1': 'Donor F1',
        'acceptor_topk': 'Acceptor Top-1',
        # 'acceptor_auprc': 'Acceptor AUPRC',
        # 'acceptor_auroc': 'Acceptor AUROC',
        # 'acceptor_accuracy': 'Acceptor Accuracy',
        # 'acceptor_precision': 'Acceptor Precision',
        # 'acceptor_recall': 'Acceptor Recall',
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
    n_cols = n_metrics // 2
    n_rows = 2  # Multiply by 2 for donor and acceptor rows
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharey=False)
    # n_cols = 5
    # n_rows = math.ceil(n_metrics / n_cols)
    # fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows), sharey=False)
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

    plt.savefig(f"{output_dir}/combined_metrics_{species}_selected_topk_f1.png", dpi=300)


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
    
    
    ##############################################
    # For each metric [Merge each flanking size]
    ##############################################
    # Initialize lists for combined metrics across all flanking sizes
    combined_keras_values = []
    combined_openspliceai_values = []
    percentage_improvements = []
    for metric in metrics_keras.keys():
        if "auprc" not in metric:
            continue
        for flanking_size in flanking_sizes:
            # Collect values for the current metric and flanking size for both models
            keras_values = [value for idx, value, fs in metrics_keras[metric] if fs == flanking_size]
            openspliceai_values = [value for idx, value, fs in metrics_pytorch[metric] if fs == flanking_size]

            # Add these values to the combined lists
            combined_keras_values.extend(keras_values)
            combined_openspliceai_values.extend(openspliceai_values)

            # Calculate the mean for each model
            mean_keras = np.mean(keras_values) if keras_values else np.nan
            mean_openspliceai = np.mean(openspliceai_values) if openspliceai_values else np.nan

            # Calculate percentage improvement
            if mean_keras != 0:
                improvement = ((mean_openspliceai - mean_keras) / mean_keras) * 100
                percentage_improvements.append(improvement)
            else:
                print(f"Mean for SpliceAI-Keras is zero for metric {metric} at flanking size {flanking_size}, skipping.")

    print("len(combined_keras_values): ", len(combined_keras_values))   
    print("len(combined_openspliceai_values): ", len(combined_openspliceai_values))
    print("len(percentage_improvements): ", len(percentage_improvements))

    # # Ensure both lists are the same length
    # assert len(combined_keras_values) == len(combined_openspliceai_values), "Lists must be of equal length"

    # print("=====================================")
    # # Calculate overall average percentage improvement
    # overall_average_improvement = np.mean(percentage_improvements) if percentage_improvements else np.nan
    # print(f"Overall average percentage improvement across all flanking sizes: {overall_average_improvement}%")

    # # Paired t-test
    # t_stat, p_value_ttest = ttest_rel(combined_openspliceai_values, combined_keras_values)
    # print(f"Paired t-test results on combined data: t-statistic = {t_stat}, p-value = {p_value_ttest}")

    # # Wilcoxon Signed-Rank Test (if non-parametric test is preferred)
    # statistic, p_value_wilcoxon = wilcoxon(combined_openspliceai_values, combined_keras_values)
    # print(f"Wilcoxon test results on combined data: statistic = {statistic}, p-value = {p_value_wilcoxon}")
    # print("=====================================")


    # ##############################################
    # # For each metric and flanking size [Separate each flanking size]
    # ##############################################
    # for metric in metrics_keras.keys():
    #     if "auprc" not in metric:
    #         continue
    #     for flanking_size in flanking_sizes:
    #         keras_values = [value for idx, value, fs in metrics_keras[metric] if fs == flanking_size]
    #         openspliceai_values = [value for idx, value, fs in metrics_pytorch[metric] if fs == flanking_size]

    #         # Ensure both lists are aligned by random seed
    #         keras_values_sorted = [x for _, x in sorted(zip(random_seeds, keras_values))]
    #         openspliceai_values_sorted = [x for _, x in sorted(zip(random_seeds, openspliceai_values))]

    #         print(f"\t{flanking_size}; {metric}: Length of openspliceai_values_sorted: {len(openspliceai_values_sorted)}")
    #         print(f"\t{flanking_size}; {metric}: Length of keras_values_sorted: {len(keras_values_sorted)}")

    #         # Perform statistical tests
    #         t_stat, p_value = ttest_rel(openspliceai_values_sorted, keras_values_sorted)
    #         print(f"{metric} at flanking size {flanking_size}: t-statistic = {t_stat}, p-value = {p_value}")

    #         # Calculate effect size
    #         effect_size = cohens_d(openspliceai_values_sorted, keras_values_sorted)
    #         print(f"{metric} at flanking size {flanking_size}: Cohen's d = {effect_size}")

    #         # Calculate percentage improvement
    #         mean_keras = np.mean(keras_values_sorted)
    #         mean_openspliceai = np.mean(openspliceai_values_sorted)
    #         percentage_improvement = ((mean_openspliceai - mean_keras) / mean_keras) * 100 if mean_keras != 0 else np.nan
    #         print(f"{metric} at flanking size {flanking_size}: Percentage improvement = {percentage_improvement}%")
    #         print("=====================================\n")

    # print("Metrics across SpliceAI-Keras:", metrics_keras)
    # print("Metrics across OpenSpliceAI:", metrics_pytorch)

    # # Calculate performance improvement for both Keras and PyTorch models
    # improvement_keras = calculate_improvement(metrics_keras, flanking_sizes)
    # improvement_pytorch = calculate_improvement(metrics_pytorch, flanking_sizes)

    # # Log the improvements
    # print("** Performance improvement for SpliceAI-Keras:")
    # for metric, improvements in improvement_keras.items():
    #     print(f"\t {metric} improvements:", improvements)

    # print("** Performance improvement for OpenSpliceAI:")
    # for metric, improvements in improvement_pytorch.items():
    #     print(f"\t {metric} improvements:", improvements)

    plot_metrics_with_error_bars(
        args.output_dir, metrics_keras, metrics_pytorch, flanking_sizes, 
        args.species, args.experiment
    )

if __name__ == "__main__":
    main()
