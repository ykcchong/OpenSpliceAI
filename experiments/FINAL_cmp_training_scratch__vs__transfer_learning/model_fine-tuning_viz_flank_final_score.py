import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

    # Read data for all seeds
    for random_seed in range(5):
        scratch_path = f'{base_path}/train_outdir/FINAL/{specie}/flanking_{condition}/SpliceAI_{specie}_train_{condition}_{random_seed}_rs1{random_seed}/{random_seed}/LOG/TEST'
        finetune_path = f'{base_path}/fine-tune_outdir/{specie}/flanking_{condition}/SpliceAI_human_{specie}_fine-tune_{condition}_{random_seed}_rs1{random_seed}/{random_seed}/LOG/TEST'

        scratch_data = process_directory(scratch_path)
        finetune_data = process_directory(finetune_path)

        scratch_data_list.append(scratch_data)
        finetune_data_list.append(finetune_data)

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

    return (averaged_scratch, std_scratch), (averaged_finetune, std_finetune)


def plot_first_last_epoch_comparison(metric_first_epoch_data_scratch, metric_first_epoch_data_finetune,
                                     metric_first_epoch_std_scratch, metric_first_epoch_std_finetune,
                                     metric_last_epoch_data_scratch, metric_last_epoch_data_finetune,
                                     metric_last_epoch_std_scratch, metric_last_epoch_std_finetune,
                                     metrics, metrics_name, specie):
    import matplotlib.pyplot as plt

    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle(f'Splice Site Prediction Metrics for {specie.capitalize()}', fontsize=22, y=0.95)

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
                            label='Scratch (1st epoch)', color=scratch_color_first,
                            **scratch_style_first, markersize=6, linewidth=2)

        # Plot last epoch scratch
        line2 = ax.errorbar(x_positions, y_last_scratch, yerr=std_last_scratch,
                            label='Scratch (10th epoch)', color=scratch_color,
                            **scratch_style_last, markersize=6, linewidth=2)

        # Plot first epoch finetune
        line3 = ax.errorbar(x_positions, y_first_finetune, yerr=std_first_finetune,
                            label='Transfer-learning (1st epoch)', color=finetune_color_first,
                            **finetune_style_first, markersize=6, linewidth=2)

        # Plot last epoch finetune
        line4 = ax.errorbar(x_positions, y_last_finetune, yerr=std_last_finetune,
                            label='Transfer-learning (10th epoch)', color=finetune_color,
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
    fig.legend(handles, labels, loc='lower center', fontsize=16, ncol=4, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])


    # # Create a single legend for all subplots
    # fig.legend(handles, labels, loc='lower center', fontsize=12, ncol=2)

    # # Adjust layout to accommodate the legend
    # plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust the top to make room for the legend

    plt.savefig(f'viz_flanking/{specie}_flanking_size_first_last_epoch_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    base_path = '/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results'
    conditions = ['80', '400', '2000', '10000']
    # species_ls = ['mouse', 'honeybee', 'zebrafish', 'arabidopsis']
    species_ls = ['arabidopsis']

    metrics = ['acceptor_topk', 'acceptor_f1', 'acceptor_auprc',
               'donor_topk', 'donor_f1', 'donor_auprc']

    metrics_name = ['Acceptor Top-1', 'Acceptor F1', 'Acceptor AUPRC',
                    'Donor Top-1', 'Donor F1', 'Donor AUPRC']

    os.makedirs('viz_flanking', exist_ok=True)
    for species in species_ls:
        # Initialize data structures to collect first and last epoch data
        metric_first_epoch_data_scratch = {metric: {} for metric in metrics}
        metric_first_epoch_data_finetune = {metric: {} for metric in metrics}
        metric_first_epoch_std_scratch = {metric: {} for metric in metrics}
        metric_first_epoch_std_finetune = {metric: {} for metric in metrics}
        metric_last_epoch_data_scratch = {metric: {} for metric in metrics}
        metric_last_epoch_data_finetune = {metric: {} for metric in metrics}
        metric_last_epoch_std_scratch = {metric: {} for metric in metrics}
        metric_last_epoch_std_finetune = {metric: {} for metric in metrics}

        for condition in conditions:
            # Average and std over seeds
            (avg_scratch, std_scratch), (avg_finetune, std_finetune) = average_metrics_over_seeds(base_path, species, condition, metrics)
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

            print(f'Finished processing {species} condition {condition}.')

        # Plot comparison for all conditions
        plot_first_last_epoch_comparison(metric_first_epoch_data_scratch, metric_first_epoch_data_finetune,
                                         metric_first_epoch_std_scratch, metric_first_epoch_std_finetune,
                                         metric_last_epoch_data_scratch, metric_last_epoch_data_finetune,
                                         metric_last_epoch_std_scratch, metric_last_epoch_std_finetune,
                                         metrics, metrics_name, species)

    print("Comparison complete. Results saved in PNG files for each species.")

if __name__ == "__main__":
    main()
