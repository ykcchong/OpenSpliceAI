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

def plot_all_conditions_as_subplots(data_dict, std_dict, metrics, metrics_name, specie, specie_title):
    """
    Create one figure that includes all condition-metric subplots.
    The grid will have 8 rows and 3 columns.
    (This works well when you have 4 conditions and 6 metrics, i.e. 24 total subplots.)
    """
    # Sort conditions numerically
    conditions_sorted = sorted(data_dict.keys(), key=lambda x: int(x))
    
    # Create a list of all (condition, metric, metric_title) tuples.
    # The order is: for each condition (in sorted order), iterate over each metric.
    subplot_keys = []
    for condition in conditions_sorted:
        for idx, metric in enumerate(metrics):
            subplot_keys.append((condition, metric, metrics_name[idx]))
    
    total_subplots = len(subplot_keys)
    
    # Fixed grid: 8 rows x 3 columns = 24 subplots.
    n_rows, n_cols = 8, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2.5))
    axes = axes.flatten()  # Flatten to simplify indexing.
    
    for idx, ax in enumerate(axes):
        if idx < total_subplots:
            condition, metric, metric_title = subplot_keys[idx]
            scratch_data, transfer_data = data_dict[condition]
            scratch_std, transfer_std = std_dict[condition]
            
            # Check if the metric exists in the data for the condition
            if metric not in scratch_data or metric not in transfer_data:
                print(f"Metric '{metric}' not found for condition '{condition}'. Skipping subplot...")
                ax.set_visible(False)
                continue
            
            # Determine epochs based on the sequence length.
            epochs = range(1, len(scratch_data[metric]) + 1)
            
            # Plot Scratch-trained data (blue, dashed, circle markers)
            ax.errorbar(
                epochs,
                scratch_data[metric],
                yerr=scratch_std[metric],
                color='#377eb8',
                linestyle='--',
                marker='o',
                markersize=4,
                linewidth=1,
                label='Scratch-trained'
            )
            
            # Plot Transfer-trained data (red, solid, square markers)
            ax.errorbar(
                epochs,
                transfer_data[metric],
                yerr=transfer_std[metric],
                color='#e41a1c',
                linestyle='-',
                marker='s',
                markersize=4,
                linewidth=1,
                label='Transfer-trained'
            )
            
            ax.set_title(f'Flank {condition} - {metric_title}', fontsize=12)
            ax.set_ylim(0.2, 1)
            max_epoch = len(scratch_data[metric])
            ax.set_xticks(np.linspace(1, max_epoch, min(10, max_epoch), dtype=int))
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.legend(fontsize=8)
        else:
            # Hide any extra axes that aren't used.
            ax.set_visible(False)
    
    fig.suptitle(f'Splice Site Prediction Metrics for {specie_title}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = f'viz/{specie}_all_conditions_subplots_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined subplot figure for {specie} at {output_path}")


def plot_comparison_per_condition(data_dict, std_dict, metrics, metrics_name, specie, specie_title):
    """
    For each condition (flanking size), create a separate plot with one subplot per metric.
    In each subplot, plot the Scratch-trained (blue dashed line with circle markers) and
    Transfer-trained (red solid line with square markers) curves.
    """
    # Sort conditions numerically (assuming conditions are numeric strings)
    for condition in sorted(data_dict.keys(), key=lambda x: int(x)):
        # Unpack the averaged metrics and standard deviations for the condition
        scratch_data, transfer_data = data_dict[condition]
        scratch_std, transfer_std = std_dict[condition]
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 4, 4.7), squeeze=False)
        # When using subplots with 1 row, extract the list of axes:
        axes = axes[0]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # If the metric is missing, hide the subplot.
            if metric not in scratch_data or metric not in transfer_data:
                print(f"Metric '{metric}' not found for condition '{condition}'. Skipping subplot...")
                ax.set_visible(False)
                continue
            
            # Determine the epochs based on the length of the metric sequence.
            epochs = range(1, len(scratch_data[metric]) + 1)
            
            # Plot Scratch-trained data (blue, dashed, circle markers)
            ax.errorbar(
                epochs,
                scratch_data[metric],
                yerr=scratch_std[metric],
                color='#377eb8',
                linestyle='--',
                marker='o',
                markersize=4,
                linewidth=1,
                label='Scratch-trained'
            )
            
            # Plot Transfer-trained data (red, solid, square markers)
            ax.errorbar(
                epochs,
                transfer_data[metric],
                yerr=transfer_std[metric],
                color='#e41a1c',
                linestyle='-',
                marker='s',
                markersize=4,
                linewidth=1,
                label='Transfer-trained'
            )
            
            ax.set_title(metrics_name[i], fontsize=16)
            ax.set_xlabel('Epoch', fontsize=14)
            ax.set_ylabel('Value', fontsize=14)
            ax.set_ylim(0.2, 1)  # Adjust if necessary based on your metric range
            max_epoch = len(scratch_data[metric])
            ax.set_xticks(np.linspace(1, max_epoch, min(10, max_epoch), dtype=int))
            ax.legend(fontsize=10)
        
        # Overall figure title includes the species and flanking size condition.
        fig.suptitle(
            f'Splice Site Prediction Metrics for {specie_title} (Flank: {condition})',
            fontsize=22,
            y=1.02
        )
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'viz/{specie}_{condition}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def calculate_auc_over_epochs(metric_data):
    """
    Calculate the AUC using the trapezoidal rule for each metric data sequence.
    """
    return np.trapz(metric_data, dx=1)  # dx=1 assumes uniform spacing (epochs 1, 2, 3, ...)

def compute_average_auc(data_dict, metrics):
    """
    Compute the average AUC for scratch and finetune data across different conditions (flanking sequence sizes).
    """
    auc_results = {}
    for metric in metrics:
        scratch_aucs = []
        finetune_aucs = []
        
        for condition, (scratch_data, finetune_data) in data_dict.items():
            # Check if the metric exists for both scratch and finetune data
            if metric in scratch_data and metric in finetune_data:
                # Calculate AUC for scratch and finetune
                auc_scratch = calculate_auc_over_epochs(scratch_data[metric])
                auc_finetune = calculate_auc_over_epochs(finetune_data[metric])
                
                # Append AUCs for each condition
                scratch_aucs.append(auc_scratch)
                finetune_aucs.append(auc_finetune)

        # Calculate average AUC across conditions for the metric
        avg_scratch_auc = np.mean(scratch_aucs) if scratch_aucs else None
        avg_finetune_auc = np.mean(finetune_aucs) if finetune_aucs else None
        auc_improvement = avg_finetune_auc - avg_scratch_auc if avg_scratch_auc is not None and avg_finetune_auc is not None else None
        
        auc_results[metric] = {
            "average_scratch_auc": avg_scratch_auc,
            "average_finetune_auc": avg_finetune_auc,
            "average_auc_improvement": auc_improvement
        }

    return auc_results

def main():
    base_path = '/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results'
    conditions = ['80', '400', '2000', '10000']
    species_ls = ['arabidopsis', 'mouse', 'honeybee', 'zebrafish']
    species_titles = [r'$\mathit{Arabidopsis}$', "Mouse", "Honeybee", "Zebrafish"]

    # For this example we use two metrics.
    # metrics = ['acceptor_topk', 'donor_topk']
    # metrics_name = ['Acceptor Top-1', 'Donor Top-1']


    # metrics = ['acceptor_topk', 'donor_topk', 'acceptor_f1', 'donor_f1'
    #            'acceptor_auprc', 'donor_auprc']

    # metrics_name = ['Acceptor Top-1', 'Donor Top-1', 'Acceptor F1', 'Donor F1', 
    #                 'Acceptor AUPRC', 'Donor AUPRC']


    metrics = ['acceptor_topk', 'acceptor_f1', 'acceptor_auprc',
               'donor_topk', 'donor_f1', 'donor_auprc']

    metrics_name = ['Acceptor Top-1', 'Acceptor F1', 'Acceptor AUPRC',
                    'Donor Top-1', 'Donor F1', 'Donor AUPRC']


    os.makedirs('viz', exist_ok=True)
    for idx, species in enumerate(species_ls):
        data_dict = {}
        std_dict = {}
        combined_df = pd.DataFrame()

        for condition in conditions:
            # Average and standard deviation calculations over seeds.
            (avg_scratch, std_scratch), (avg_finetune, std_finetune) = average_metrics_over_seeds(
                base_path, species, condition, metrics)
            print(f'Finished processing {species} condition {condition}.')
            data_dict[condition] = (avg_scratch, avg_finetune)
            std_dict[condition] = (std_scratch, std_finetune)
            print(f'Finished processing {species} condition {condition}.')

        # Plot separate figures for each condition.
        plot_comparison_per_condition(data_dict, std_dict, metrics, metrics_name,
                                      species, species_titles[idx])

        # New: Plot a single figure with subplots for all conditions.
        plot_all_conditions_as_subplots(data_dict, std_dict, metrics, metrics_name,
                                        species, species_titles[idx])

        # Optionally, save the combined DataFrame (if populated).
        combined_df.to_csv(f'viz/{species}_all_conditions_comparison.csv', index=False)

        # Compute and print average AUC values across different conditions.
        avg_auc_results = compute_average_auc(data_dict, metrics)
        print(f"Average AUC across different flanking sizes for {species}")
        for metric, auc_data in avg_auc_results.items():
            if "_topk" not in metric:
                continue
            print(f"  Metric: {metric}")
            print(f"    Average Scratch AUC: {auc_data['average_scratch_auc']:.4f}")
            print(f"    Average Fine-tune AUC: {auc_data['average_finetune_auc']:.4f}")
            print(f"    Average AUC Improvement: {auc_data['average_auc_improvement']:.4f}")
        print("======================")
    print("Comparison complete. Results saved in CSV files and PNG files for each species.")

if __name__ == "__main__":
    main()
