import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

def read_data(file_path):
    with open(file_path, 'r') as f:
        return [float(line.strip()) for line in f]

def process_directory(dir_path):
    data = {}
    # Check if directory exists
    if not os.path.exists(dir_path):
        print(f'Directory {dir_path} does not exist. Skipping...')
        return data
    for filename in os.listdir(dir_path):
        if filename.endswith('.txt'):
            if filename in ['donor_topk_all.txt', 'acceptor_topk_all.txt', 'loss_every_update.txt']:
                continue
            file_path = os.path.join(dir_path, filename)
            # Check if file exists
            if not os.path.exists(file_path):
                print(f'File {file_path} does not exist. Skipping...')
                continue
            print(f'Reading {filename}...')
            metric_name = os.path.splitext(filename)[0]
            data[metric_name] = read_data(file_path)
    return data


def average_metrics_over_seeds(base_path, specie, condition, metrics):
    scratch_data_list = []
    finetune_data_list = []
    per_seed_scratch = {metric: [] for metric in metrics}
    per_seed_finetune = {metric: [] for metric in metrics}

    # Read data for all seeds
    for random_seed in range(5):
        scratch_path = f'{base_path}/train_outdir/FINAL/{specie}/flanking_{condition}/SpliceAI_{specie}_train_{condition}_{random_seed}_rs1{random_seed}/{random_seed}/LOG/TEST'
        finetune_path = f'{base_path}/fine-tune_outdir/{specie}/flanking_{condition}/SpliceAI_human_{specie}_fine-tune_{condition}_{random_seed}_rs1{random_seed}/{random_seed}/LOG/TEST'

        scratch_data = process_directory(scratch_path)
        finetune_data = process_directory(finetune_path)

        scratch_data_list.append(scratch_data)
        finetune_data_list.append(finetune_data)

        # Collect per-seed data per metric
        for metric in metrics:
            if metric in scratch_data:
                per_seed_scratch[metric].append(scratch_data[metric])
            else:
                per_seed_scratch[metric].append([])
            if metric in finetune_data:
                per_seed_finetune[metric].append(finetune_data[metric])
            else:
                per_seed_finetune[metric].append([])

    # Average and standard deviation calculations
    averaged_scratch = {}
    std_scratch = {}
    averaged_finetune = {}
    std_finetune = {}

    for metric in metrics:
        # Collect all seeds' data for the specific metric
        scratch_metric_data = [data[metric] for data in scratch_data_list if metric in data]
        finetune_metric_data = [data[metric] for data in finetune_data_list if metric in data]

        # Skip if no data is available for the metric
        if not scratch_metric_data and not finetune_metric_data:
            print(f"No data available for metric {metric}. Skipping...")
            continue

        # Find the maximum length among sequences for the metric
        max_length = max([len(seq) for seq in scratch_metric_data + finetune_metric_data])

        # Pad sequences to the maximum length with NaN
        padded_scratch_data = [np.pad(seq, (0, max_length - len(seq)), constant_values=np.nan) for seq in scratch_metric_data]
        padded_finetune_data = [np.pad(seq, (0, max_length - len(seq)), constant_values=np.nan) for seq in finetune_metric_data]

        # Calculate average and standard deviation while ignoring NaNs
        if padded_scratch_data:
            averaged_scratch[metric] = np.nanmean(padded_scratch_data, axis=0)
            std_scratch[metric] = np.nanstd(padded_scratch_data, axis=0)
        if padded_finetune_data:
            averaged_finetune[metric] = np.nanmean(padded_finetune_data, axis=0)
            std_finetune[metric] = np.nanstd(padded_finetune_data, axis=0)

    # Return both the averages and the per-seed data
    return (averaged_scratch, std_scratch), (averaged_finetune, std_finetune), per_seed_scratch, per_seed_finetune


def plot_first_last_epoch_comparison(metric_first_epoch_data_scratch, metric_first_epoch_data_finetune,
                                     metric_first_epoch_std_scratch, metric_first_epoch_std_finetune,
                                     metric_last_epoch_data_scratch, metric_last_epoch_data_finetune,
                                     metric_last_epoch_std_scratch, metric_last_epoch_std_finetune,
                                     metrics, metrics_name, specie, specie_title):
    import matplotlib.pyplot as plt

    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 5 * n_rows))
    fig.suptitle(f'Splice Site Prediction Metrics for {specie_title.capitalize()}', fontsize=22, y=0.95)

    # Define colors and styles
    scratch_color = '#377eb8'  # Blue
    scratch_color_first = '#7fadd1'
    finetune_color = '#e41a1c'  # Red
    finetune_color_first = '#ed6d6d'

    scratch_style_first = {'linestyle': '--', 'marker': 'o'}
    scratch_style_last = {'linestyle': '-', 'marker': 'o'}
    finetune_style_first = {'linestyle': '--', 'marker': 's'}
    finetune_style_last = {'linestyle': '-', 'marker': 's'}

    # Initialize lists to collect legend handles and labels
    handles = []
    labels = []

    for i, metric in enumerate(metrics):
        metric_name = metrics_name[i]
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]

        # Get conditions and convert to integers
        conditions = list(metric_first_epoch_data_scratch[metric].keys())
        flanking_sizes = [int(cond) for cond in conditions]

        # Sort flanking_sizes and conditions accordingly
        sorted_pairs = sorted(zip(flanking_sizes, conditions))
        flanking_sizes_sorted = [x[0] for x in sorted_pairs]
        conditions_sorted = [x[1] for x in sorted_pairs]

        # Map flanking sizes to categorical indices
        x_positions = range(len(flanking_sizes_sorted))

        # Get y-values and stds for first and last epochs, scratch and finetune
        y_first_scratch = [metric_first_epoch_data_scratch[metric][cond] for cond in conditions_sorted]
        y_last_scratch = [metric_last_epoch_data_scratch[metric][cond] for cond in conditions_sorted]
        std_first_scratch = [metric_first_epoch_std_scratch[metric][cond] for cond in conditions_sorted]
        std_last_scratch = [metric_last_epoch_std_scratch[metric][cond] for cond in conditions_sorted]

        y_first_finetune = [metric_first_epoch_data_finetune[metric][cond] for cond in conditions_sorted]
        y_last_finetune = [metric_last_epoch_data_finetune[metric][cond] for cond in conditions_sorted]
        std_first_finetune = [metric_first_epoch_std_finetune[metric][cond] for cond in conditions_sorted]
        std_last_finetune = [metric_last_epoch_std_finetune[metric][cond] for cond in conditions_sorted]

        # Plot first epoch scratch
        line1 = ax.errorbar(x_positions, y_first_scratch, yerr=std_first_scratch,
                            label='Scratch-trained (1st epoch)', color=scratch_color_first,
                            **scratch_style_first, markersize=6, linewidth=2)

        # Plot last epoch scratch
        line2 = ax.errorbar(x_positions, y_last_scratch, yerr=std_last_scratch,
                            label='Scratch-trained (10th epoch)', color=scratch_color,
                            **scratch_style_last, markersize=6, linewidth=2)

        # Plot first epoch finetune
        line3 = ax.errorbar(x_positions, y_first_finetune, yerr=std_first_finetune,
                            label='Transfer-learned (1st epoch)', color=finetune_color_first,
                            **finetune_style_first, markersize=6, linewidth=2)

        # Plot last epoch finetune
        line4 = ax.errorbar(x_positions, y_last_finetune, yerr=std_last_finetune,
                            label='Transfer-learned (10th epoch)', color=finetune_color,
                            **finetune_style_last, markersize=6, linewidth=2)

        # Collect handles and labels for the legend
        if i == 0:
            handles.extend([line1, line2, line3, line4])
            labels.extend([line1.get_label(), line2.get_label(), line3.get_label(), line4.get_label()])

        ax.set_title(metric_name, fontsize=16)
        ax.set_xlabel('Flanking Size', fontsize=14)
        ax.set_ylabel('Value', fontsize=14)
        ax.set_ylim(0.2, 1)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Set x-axis ticks to categorical positions
        ax.set_xticks(x_positions)
        ax.set_xticklabels(flanking_sizes_sorted)

    # Adjust layout to prevent overlap
    plt.subplots_adjust(wspace=0.3, hspace=0.6, top=0.88)

    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    # Create a single legend for all subplots at the bottom
    # fig.legend(handles, labels, loc='lower center', fontsize=16, ncol=4, bbox_to_anchor=(0.5, -0.05))

    # Create a single legend for all subplots
    fig.legend(handles, labels, loc='lower center', fontsize=8, ncol=4)


    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    # # Adjust layout to accommodate the legend
    # plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust the top to make room for the legend

    plt.savefig(f'viz_flanking/{specie}_flanking_size_first_last_epoch_comparison_selected.png', dpi=300, bbox_inches='tight')
    plt.close()


def calculate_improvement_statistics(metrics, averaged_scratch, averaged_finetune):
    print("averaged_scratch: ", averaged_scratch)
    print("averaged_finetune: ", averaged_finetune)
    combined_scratch_values = []
    combined_finetune_values = []
    percentage_improvements = []

    # Iterate over each metric
    for metric in metrics:
        if metric in averaged_scratch and metric in averaged_finetune:
            # Retrieve scratch and finetune data for each flanking size in the metric
            for flanking_size in averaged_scratch[metric]:
                scratch_values = averaged_scratch[metric]
                finetune_values = averaged_finetune[metric]

                # Collect values for the current metric and flanking size for both models
                combined_scratch_values.extend(scratch_values)
                combined_finetune_values.extend(finetune_values)

                # Calculate the mean for each model
                mean_scratch = np.mean(scratch_values) if scratch_values else np.nan
                mean_finetune = np.mean(finetune_values) if finetune_values else np.nan

                # Calculate percentage improvement
                if mean_scratch != 0:
                    improvement = ((mean_finetune - mean_scratch) / mean_scratch) * 100
                    percentage_improvements.append(improvement)
                else:
                    print(f"Mean for scratch is zero for metric {metric}, flanking size {flanking_size}. Skipping...")

    # Ensure both lists are the same length for statistical tests
    assert len(combined_scratch_values) == len(combined_finetune_values), "Lists must be of equal length"

    # Calculate overall average percentage improvement
    overall_average_improvement = np.mean(percentage_improvements) if percentage_improvements else np.nan
    print("=====================================")
    print(f"Overall average percentage improvement across all flanking sizes: {overall_average_improvement}%")

    # Paired t-test
    t_stat, p_value_ttest = ttest_rel(combined_finetune_values, combined_scratch_values)
    print(f"Paired t-test results on combined data: t-statistic = {t_stat}, p-value = {p_value_ttest}")

    # Wilcoxon Signed-Rank Test
    statistic, p_value_wilcoxon = wilcoxon(combined_finetune_values, combined_scratch_values)
    print(f"Wilcoxon test results on combined data: statistic = {statistic}, p-value = {p_value_wilcoxon}")
    print("=====================================")


def main():
    base_path = '/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results'
    conditions = ['80', '400', '2000', '10000']
    # species_ls = ['mouse', 'honeybee', 'zebrafish', 'arabidopsis']
    species_ls = ['arabidopsis', 'mouse', 'honeybee', 'zebrafish']
    species_titles = ['Thale cress', "Mouse", "Honeybee", "Zebrafish"]

    # metrics = ['acceptor_topk', 'acceptor_f1', 'acceptor_auprc',
    #            'donor_topk', 'donor_f1', 'donor_auprc']

    # metrics_name = ['Acceptor Top-1', 'Acceptor F1', 'Acceptor AUPRC',
    #                 'Donor Top-1', 'Donor F1', 'Donor AUPRC']

    metrics = ['acceptor_topk', 'donor_topk']

    metrics_name = ['Acceptor Top-1', 'Donor Top-1']


    os.makedirs('viz_flanking', exist_ok=True)
    for idx, species in enumerate(species_ls):
        # Initialize data structures to collect first and last epoch data
        metric_first_epoch_data_scratch = {metric: {} for metric in metrics}
        metric_first_epoch_data_finetune = {metric: {} for metric in metrics}
        metric_first_epoch_std_scratch = {metric: {} for metric in metrics}
        metric_first_epoch_std_finetune = {metric: {} for metric in metrics}
        metric_last_epoch_data_scratch = {metric: {} for metric in metrics}
        metric_last_epoch_data_finetune = {metric: {} for metric in metrics}
        metric_last_epoch_std_scratch = {metric: {} for metric in metrics}
        metric_last_epoch_std_finetune = {metric: {} for metric in metrics}

        # Initialize data structures for per-seed last epoch data
        per_seed_last_epoch_data_scratch = {metric: {condition: [] for condition in conditions} for metric in metrics}
        per_seed_last_epoch_data_finetune = {metric: {condition: [] for condition in conditions} for metric in metrics}

        # Initialize data structures to collect per-seed improvements
        per_seed_improvements_scratch = {metric: {condition: [] for condition in conditions} for metric in metrics}
        per_seed_improvements_finetune = {metric: {condition: [] for condition in conditions} for metric in metrics}

        # Initialize combined data structures for statistical tests
        combined_scratch_values_all = []
        combined_finetune_values_all = []
        percentage_improvements_all = []

        for condition in conditions:
            # Average and std over seeds
            (avg_scratch, std_scratch), (avg_finetune, std_finetune), per_seed_scratch, per_seed_finetune = average_metrics_over_seeds(base_path, species, condition, metrics)
            print(f'Finished processing {species} condition {condition}.')

            # Collect first and last epoch's data
            for metric in metrics:
                if metric in avg_scratch and metric in avg_finetune:
                    first_epoch_value_scratch = avg_scratch[metric][0]
                    std_first_epoch_scratch = std_scratch[metric][0]
                    last_epoch_value_scratch = avg_scratch[metric][-1]
                    std_last_epoch_scratch = std_scratch[metric][-1]
                    first_epoch_value_finetune = avg_finetune[metric][0]
                    std_first_epoch_finetune = std_finetune[metric][0]
                    last_epoch_value_finetune = avg_finetune[metric][-1]
                    std_last_epoch_finetune = std_finetune[metric][-1]

                    # Store values
                    metric_first_epoch_data_scratch[metric][condition] = first_epoch_value_scratch
                    metric_first_epoch_std_scratch[metric][condition] = std_first_epoch_scratch
                    metric_last_epoch_data_scratch[metric][condition] = last_epoch_value_scratch
                    metric_last_epoch_std_scratch[metric][condition] = std_last_epoch_scratch
                    metric_first_epoch_data_finetune[metric][condition] = first_epoch_value_finetune
                    metric_first_epoch_std_finetune[metric][condition] = std_first_epoch_finetune
                    metric_last_epoch_data_finetune[metric][condition] = last_epoch_value_finetune
                    metric_last_epoch_std_finetune[metric][condition] = std_last_epoch_finetune

                    # Collect per-seed last epoch data for statistical tests
                    for seed_index in range(len(per_seed_scratch[metric])):
                        scratch_values = per_seed_scratch[metric][seed_index]
                        finetune_values = per_seed_finetune[metric][seed_index]

                        # Get last epoch values
                        if scratch_values:
                            last_value_scratch = scratch_values[-1]
                        else:
                            last_value_scratch = np.nan
                        if finetune_values:
                            last_value_finetune = finetune_values[-1]
                        else:
                            last_value_finetune = np.nan

                        per_seed_last_epoch_data_scratch[metric][condition].append(last_value_scratch)
                        per_seed_last_epoch_data_finetune[metric][condition].append(last_value_finetune)

                        # Ensure we have data for both epochs
                        if scratch_values and len(scratch_values) >= 10:
                            improvement_scratch = scratch_values[9] - scratch_values[0]
                            per_seed_improvements_scratch[metric][condition].append(improvement_scratch)
                        else:
                            per_seed_improvements_scratch[metric][condition].append(np.nan)

                        if finetune_values and len(finetune_values) >= 10:
                            improvement_finetune = finetune_values[9] - finetune_values[0]
                            per_seed_improvements_finetune[metric][condition].append(improvement_finetune)
                        else:
                            per_seed_improvements_finetune[metric][condition].append(np.nan)

            print(f'Finished processing {species} condition {condition}.')

        # # Calculate improvement statistics after getting the averaged values
        # calculate_improvement_statistics(metrics, metric_last_epoch_data_scratch, metric_last_epoch_data_finetune)

        # Plot comparison for all conditions
        plot_first_last_epoch_comparison(metric_first_epoch_data_scratch, metric_first_epoch_data_finetune,
                                         metric_first_epoch_std_scratch, metric_first_epoch_std_finetune,
                                         metric_last_epoch_data_scratch, metric_last_epoch_data_finetune,
                                         metric_last_epoch_std_scratch, metric_last_epoch_std_finetune,
                                         metrics, metrics_name, species, species_titles[idx])

        
        # Now perform statistical comparisons between the improvements of scratch and finetune
        from scipy.stats import ttest_rel, wilcoxon

        print("=====================================")
        print(f"Improvement from Epoch 1 to 10 for {species.capitalize()}:")

        for metric_idx, metric in enumerate(metrics):
            print(f"\nMetric: {metrics_name[metric_idx]} ({metric})")
            combined_improvements_scratch = []
            combined_improvements_finetune = []

            for condition in conditions:
                improvements_scratch = per_seed_improvements_scratch[metric][condition]
                improvements_finetune = per_seed_improvements_finetune[metric][condition]

                # Remove NaN values
                valid_indices = [i for i in range(len(improvements_scratch)) if not np.isnan(improvements_scratch[i]) and not np.isnan(improvements_finetune[i])]
                improvements_scratch = [improvements_scratch[i] for i in valid_indices]
                improvements_finetune = [improvements_finetune[i] for i in valid_indices]

                combined_improvements_scratch.extend(improvements_scratch)
                combined_improvements_finetune.extend(improvements_finetune)

                # Calculate average improvements
                avg_improvement_scratch = np.mean(improvements_scratch) if improvements_scratch else np.nan
                avg_improvement_finetune = np.mean(improvements_finetune) if improvements_finetune else np.nan

                print(f"Condition {condition}:")
                print(f"  Average Improvement Scratch: {avg_improvement_scratch:.4f}")
                print(f"  Average Improvement Fine-tune: {avg_improvement_finetune:.4f}")

            # Perform statistical tests across all conditions
            if combined_improvements_scratch and combined_improvements_finetune:
                # Paired t-test
                t_stat, p_value_ttest = ttest_rel(combined_improvements_finetune, combined_improvements_scratch)
                # Wilcoxon Signed-Rank Test
                statistic, p_value_wilcoxon = wilcoxon(combined_improvements_finetune, combined_improvements_scratch)

                print("\nCombined Conditions:")
                print(f"  Paired t-test: t-statistic = {t_stat:.4f}, p-value = {p_value_ttest:.4f}")
                print(f"  Wilcoxon test: statistic = {statistic:.4f}, p-value = {p_value_wilcoxon:.4f}")
            else:
                print("No valid data for statistical tests.")

        print("=====================================")

    print("Comparison complete.")


    #     # Calculate improvement statistics
    #     from scipy.stats import ttest_rel, wilcoxon

    #     print("=====================================")
    #     print(f"Improvement Statistics for {species.capitalize()}:")
    #     overall_percentage_improvements = []
    #     combined_scratch_values = []
    #     combined_finetune_values = []

    #     for metric in metrics:
    #         metric_percentage_improvements = []
    #         metric_scratch_values = []
    #         metric_finetune_values = []

    #         for condition in conditions:
    #             scratch_values = per_seed_last_epoch_data_scratch[metric][condition]
    #             finetune_values = per_seed_last_epoch_data_finetune[metric][condition]

    #             for value_scratch, value_finetune in zip(scratch_values, finetune_values):
    #                 if not np.isnan(value_scratch) and not np.isnan(value_finetune) and value_scratch != 0:
    #                     percentage_improvement = ((value_finetune - value_scratch) / value_scratch) * 100
    #                     metric_percentage_improvements.append(percentage_improvement)
    #                     metric_scratch_values.append(value_scratch)
    #                     metric_finetune_values.append(value_finetune)
    #                     combined_scratch_values_all.append(value_scratch)
    #                     combined_finetune_values_all.append(value_finetune)
    #                     percentage_improvements_all.append(percentage_improvement)
    #                 else:
    #                     print(f"Invalid values for metric {metric}, condition {condition}, skipping.")

    #         # Compute average percentage improvement and perform statistical tests for the metric
    #         if metric_percentage_improvements:
    #             average_improvement = np.mean(metric_percentage_improvements)
    #             overall_percentage_improvements.extend(metric_percentage_improvements)
    #             combined_scratch_values.extend(metric_scratch_values)
    #             combined_finetune_values.extend(metric_finetune_values)

    #             # Paired t-test
    #             t_stat, p_value_ttest = ttest_rel(metric_finetune_values, metric_scratch_values)
    #             # Wilcoxon Signed-Rank Test
    #             statistic, p_value_wilcoxon = wilcoxon(metric_finetune_values, metric_scratch_values)

    #             print(f"Metric {metric}:")
    #             print(f"  Average percentage improvement: {average_improvement:.2f}%")
    #             print(f"  Paired t-test: t-statistic = {t_stat:.4f}, p-value = {p_value_ttest:.4f}")
    #             print(f"  Wilcoxon test: statistic = {statistic:.4f}, p-value = {p_value_wilcoxon:.4f}")
    #         else:
    #             print(f"No valid data for metric {metric}")

    #     # Overall statistics across all metrics
    #     if percentage_improvements_all:
    #         overall_average_improvement = np.mean(percentage_improvements_all)
    #         # Paired t-test
    #         t_stat_all, p_value_ttest_all = ttest_rel(combined_finetune_values_all, combined_scratch_values_all)
    #         # Wilcoxon Signed-Rank Test
    #         statistic_all, p_value_wilcoxon_all = wilcoxon(combined_finetune_values_all, combined_scratch_values_all)

    #         print("\nOverall Statistics Across All Metrics:")
    #         print(f"  Overall average percentage improvement: {overall_average_improvement:.2f}%")
    #         print(f"  Paired t-test: t-statistic = {t_stat_all:.4f}, p-value = {p_value_ttest_all:.4f}")
    #         print(f"  Wilcoxon test: statistic = {statistic_all:.4f}, p-value = {p_value_wilcoxon_all:.4f}")
    #     else:
    #         print("No valid data across all metrics.")
    #     print("=====================================")
    # print("Comparison complete. Results saved in PNG files for each species.")

if __name__ == "__main__":
    main()
