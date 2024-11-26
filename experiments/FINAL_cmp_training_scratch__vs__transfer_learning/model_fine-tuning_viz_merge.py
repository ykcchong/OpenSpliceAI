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


# def plot_comparison(data_dict, std_dict, metrics, specie):
#     n_metrics = len(metrics)
#     n_cols = 5
#     n_rows = (n_metrics + n_cols - 1) // n_cols

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 5 * n_rows))
#     fig.suptitle(f'Splice site prediction metrics for {specie.capitalize()}', fontsize=16)

#     # Define a contrasting color palette
#     colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
#               '#ff7f00', '#a65628', '#f781bf', '#999999']

#     # Define the adjust_lightness function
#     def adjust_lightness(color, amount=1.0):
#         import matplotlib.colors as mc
#         import colorsys

#         try:
#             c = mc.cnames[color]
#         except KeyError:
#             c = color
#         rgb = mc.to_rgb(c)
#         h, l, s = colorsys.rgb_to_hls(*rgb)
#         l = max(0, min(1, l * amount))
#         rgb_new = colorsys.hls_to_rgb(h, l, s)
#         return rgb_new

#     for i, metric in enumerate(metrics):
#         row = i // n_cols
#         col = i % n_cols
#         ax = axes[row, col] if n_rows > 1 else axes[col]

#         plotted_anything = False  # Track if any plots are made for this metric

#         for (condition, (scratch_data, finetune_data)), (condition_std, (scratch_std, finetune_std)), color in zip(data_dict.items(), std_dict.items(), colors):
#             # Check if the metric exists in both scratch_data and finetune_data
#             if metric not in scratch_data or metric not in finetune_data:
#                 print(f"Metric '{metric}' not found for condition '{condition}'. Skipping...")
#                 continue

#             # Only calculate epochs when metric exists
#             epochs = range(1, len(scratch_data[metric]) + 1)

#             # Adjust colors for fills
#             fill_color_scratch = adjust_lightness(color, amount=1.2)  # Lighten color
#             fill_color_finetune = adjust_lightness(color, amount=0.8)  # Darken color

#             # Plot scratch data with std
#             ax.plot(epochs, scratch_data[metric], color=color, linestyle='--', label=f'Training from scratch {condition}')
#             ax.fill_between(epochs,
#                             scratch_data[metric] - scratch_std[metric],
#                             scratch_data[metric] + scratch_std[metric],
#                             color=fill_color_scratch, alpha=0.3)

#             # Plot finetune data with std
#             ax.plot(epochs, finetune_data[metric], color=color, linestyle='-', label=f'Transfer learning {condition}')
#             ax.fill_between(epochs,
#                             finetune_data[metric] - finetune_std[metric],
#                             finetune_data[metric] + finetune_std[metric],
#                             color=fill_color_finetune, alpha=0.3)

#             plotted_anything = True

#         # Add labels and legend only if any data was plotted
#         if plotted_anything:
#             ax.set_title(metric)
#             ax.set_xlabel('Epoch')
#             ax.set_ylabel('Value')
#             ax.legend(loc='best', fontsize='x-small', ncol=2)
#             ax.set_ylim(0, 1)  # Assuming metrics are between 0 and 1

#             # Set x-ticks to show fewer epoch numbers
#             max_epoch = len(epochs)
#             ax.set_xticks(np.linspace(1, max_epoch, 5, dtype=int))
#             ax.tick_params(axis='x', rotation=0)
#         else:
#             # If nothing was plotted, hide the subplot
#             ax.set_visible(False)

#     # Remove any unused subplots
#     for i in range(n_metrics, n_rows * n_cols):
#         row = i // n_cols
#         col = i % n_cols
#         fig.delaxes(axes[row, col] if n_rows > 1 else axes[col])

#     plt.tight_layout()
#     plt.savefig(f'viz/{specie}_all_conditions_comparison.png', dpi=300, bbox_inches='tight')
#     plt.close()


def plot_comparison(data_dict, std_dict, metrics, metrics_name, specie):
    import matplotlib.colors as mc
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle(f'Splice Site Prediction Metrics for {specie.capitalize()}', fontsize=22, y=0.95)

    # scratch_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
    #                   '#ff7f00', '#a65628', '#f781bf', '#999999']
    scratch_colors = ['#ed6d6d', '#7fadd1', '#8dcc8b', '#bd8dc4',
                      '#ffad5c', '#c69375', '#faaed6', '#bdbdbd']
    finetune_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                       '#ff7f00', '#a65628', '#f781bf', '#999999']
    
    # Define line styles and markers
    scratch_style = {'linestyle': '--', 'marker': 'o'}
    finetune_style = {'linestyle': '-', 'marker': 's'}

    for i, metric in enumerate(metrics):
        metric_name = metrics_name[i]
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]

        plotted_anything = False  # Track if any plots are made for this metric

        for idx, ((condition, (scratch_data, finetune_data)), (condition_std, (scratch_std, finetune_std))) in enumerate(zip(data_dict.items(), std_dict.items())):
            # Assign colors
            scratch_color = scratch_colors[idx % len(scratch_colors)]
            finetune_color = finetune_colors[idx % len(finetune_colors)]

            # Check if the metric exists
            if metric not in scratch_data or metric not in finetune_data:
                print(f"Metric '{metric}' not found for condition '{condition}'. Skipping...")
                continue

            # Only calculate epochs when metric exists
            epochs = range(1, len(scratch_data[metric]) + 1)

            # Plot scratch data with error bars
            ax.errorbar(epochs, scratch_data[metric],
                        yerr=scratch_std[metric],
                        color=scratch_color,
                        label=f'Training from scratch {condition}',
                        **scratch_style, markersize=4, linewidth=1)

            # Plot finetune data with error bars
            ax.errorbar(epochs, finetune_data[metric],
                        yerr=finetune_std[metric],
                        color=finetune_color,
                        label=f'Transfer-learning {condition}',
                        **finetune_style, markersize=4, linewidth=1)

            plotted_anything = True

        # Add labels and legend only if any data was plotted
        if plotted_anything:
            ax.set_title(metric_name, fontsize=16)
            ax.set_xlabel('Epoch', fontsize=14)
            ax.set_ylabel('Value', fontsize=14)
            ax.set_ylim(0.2, 1)  # Assuming metrics are between 0 and 1
            ax.tick_params(axis='both', which='major', labelsize=12)

            max_epoch = len(epochs)
            ax.set_xticks(np.linspace(1, max_epoch, min(10, max_epoch), dtype=int))
            ax.tick_params(axis='x', rotation=0)
        else:
            # If nothing was plotted, hide the subplot
            ax.set_visible(False)

    # Adjust layout to prevent overlap
    plt.subplots_adjust(wspace=0.3, hspace=0.6, top=0.88)

    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    # Create a single legend for all subplots at the bottom
    fig.legend(handles, labels, loc='lower center', fontsize='large', ncol=4, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(f'viz/{specie}_all_conditions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


# def plot_comparison(data_dict, std_dict, metrics, metrics_name, specie):
#     import matplotlib.colors as mc
#     n_metrics = len(metrics)
#     n_cols = 3
#     n_rows = (n_metrics + n_cols - 1) // n_cols

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
#     fig.suptitle(f'Splice Site Prediction Metrics for {specie.capitalize()}', fontsize=20)

#     # scratch_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
#     #                   '#ff7f00', '#a65628', '#f781bf', '#999999']
#     scratch_colors = ['#ed6d6d', '#7fadd1', '#8dcc8b', '#bd8dc4',
#                     '#ffad5c', '#c69375', '#faaed6', '#bdbdbd']
#     finetune_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
#                        '#ff7f00', '#a65628', '#f781bf', '#999999']
    
#     # Define line styles and markers
#     scratch_style = {'linestyle': '--', 'marker': 'o'}
#     finetune_style = {'linestyle': '-', 'marker': 's'}

#     for i, metric in enumerate(metrics):
#         metric_name = metrics_name[i]
#         row = i // n_cols
#         col = i % n_cols
#         ax = axes[row, col] if n_rows > 1 else axes[col]

#         plotted_anything = False  # Track if any plots are made for this metric

#         for idx, ((condition, (scratch_data, finetune_data)), (condition_std, (scratch_std, finetune_std))) in enumerate(zip(data_dict.items(), std_dict.items())):
#             # Assign colors
#             scratch_color = scratch_colors[idx % len(scratch_colors)]
#             finetune_color = finetune_colors[idx % len(finetune_colors)]

#             # Check if the metric exists
#             if metric not in scratch_data or metric not in finetune_data:
#                 print(f"Metric '{metric}' not found for condition '{condition}'. Skipping...")
#                 continue

#             # Only calculate epochs when metric exists
#             epochs = range(1, len(scratch_data[metric]) + 1)

#             # Plot scratch data with error bars
#             ax.errorbar(epochs, scratch_data[metric],
#                         yerr=scratch_std[metric],
#                         color=scratch_color,
#                         label=f'Scratch {condition}',
#                         **scratch_style, markersize=4, linewidth=1)

#             # Plot finetune data with error bars
#             ax.errorbar(epochs, finetune_data[metric],
#                         yerr=finetune_std[metric],
#                         color=finetune_color,
#                         label=f'Fine-tune {condition}',
#                         **finetune_style, markersize=4, linewidth=1)

#             plotted_anything = True

#         # Add labels and legend only if any data was plotted
#         if plotted_anything:
#             ax.set_title(metric_name, fontsize=16)
#             ax.set_xlabel('Epoch', fontsize=14)
#             ax.set_ylabel('Value', fontsize=14)
#             ax.set_ylim(0.3, 1)  # Assuming metrics are between 0 and 1
#             ax.tick_params(axis='both', which='major', labelsize=12)

#             # Set x-ticks to show fewer epoch numbers
#             max_epoch = len(epochs)
#             ax.set_xticks(np.linspace(1, max_epoch, min(5, max_epoch), dtype=int))
#             ax.tick_params(axis='x', rotation=0)
#         else:
#             # If nothing was plotted, hide the subplot
#             ax.set_visible(False)

#     # Adjust legend
#     handles, labels = ax.get_legend_handles_labels()
#     # Create a single legend for all subplots
#     fig.legend(handles, labels, loc='upper center', fontsize='large', ncol=4)

#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.savefig(f'viz/{specie}_all_conditions_comparison.png', dpi=300, bbox_inches='tight')
#     plt.close()



def main():
    base_path = '/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results'
    conditions = ['80', '400', '2000', '10000']
    # species_ls = ['arabidopsis', 'mouse', 'bee', 'zebrafish']
    species_ls = ['arabidopsis']
    # species_ls = ['mouse', 'honeybee', 'zebrafish', 'arabidopsis']

    # metrics = ['acceptor_topk', 'acceptor_accuracy', 'acceptor_precision',
    #            'acceptor_recall', 'acceptor_f1', 'acceptor_auprc', 
    #            'donor_topk', 'donor_accuracy', 'donor_precision', 'donor_recall', 'donor_f1', 'donor_auprc', 'accuracy']

    metrics = ['acceptor_topk', 'acceptor_f1', 'acceptor_auprc', 
               'donor_topk', 'donor_f1', 'donor_auprc']

    metrics_name = ['Acceptor Top-1', 'Acceptor F1', 'Acceptor AUPRC', 
               'Donor Top-1', 'Donor F1', 'Donor AUPRC']

    os.makedirs('viz', exist_ok=True)
    for species in species_ls:
        data_dict = {}
        std_dict = {}
        combined_df = pd.DataFrame()

        for condition in conditions:
            # Average and std over seeds
            (avg_scratch, std_scratch), (avg_finetune, std_finetune) = average_metrics_over_seeds(base_path, species, condition, metrics)
            print(f'Finished processing {species} condition {condition}.')   
            print('avg_scratch:', avg_scratch)
            print('std_scratch:', std_scratch)
            print('avg_finetune:', avg_finetune)
            print('std_finetune:', std_finetune)
            print('======================')
            data_dict[condition] = (avg_scratch, avg_finetune)
            std_dict[condition] = (std_scratch, std_finetune)

            # # Add data to combined DataFrame
            # for metric in metrics:
            #     if metric in avg_scratch:
            #         combined_df[f'{metric}_scratch_{condition}'] = avg_scratch[metric]
            #         combined_df[f'{metric}_finetune_{condition}'] = avg_finetune[metric]
            print(f'Finished processing {species} condition {condition}.')

        # Plot comparison for all conditions
        plot_comparison(data_dict, std_dict, metrics, metrics_name, species)

        # Save combined DataFrame
        combined_df.to_csv(f'viz/{species}_all_conditions_comparison.csv', index=False)

    print("Comparison complete. Results saved in CSV files and PNG files for each species.")

if __name__ == "__main__":
    main()
