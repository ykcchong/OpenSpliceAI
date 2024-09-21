import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_data(file_path):
    with open(file_path, 'r') as f:
        data = [float(line.strip()) for line in f]
    # If the length is less than 10, append 0s to make it length 10
    while len(data) < 10:
        data.append(0.0)
    return data


def process_directory(dir_path):
    data = {}
    for filename in os.listdir(dir_path):
        if filename.endswith('.txt'):
            if filename in ['donor_topk_all.txt', 'acceptor_topk_all.txt', 'loss_every_update.txt']:
                continue
            print(f'Reading {filename}...')
            metric_name = os.path.splitext(filename)[0]
            file_path = os.path.join(dir_path, filename)
            data[metric_name] = read_data(file_path)
    return data


def plot_comparison(data_dict, metrics, specie):
    n_metrics = len(metrics)
    n_cols = 5
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 5*n_rows))
    fig.suptitle(f'Splice site prediction metrics for {specie.capitalize()}', fontsize=16)

    # Define a contrasting color palette
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999']

    for i, metric in enumerate(metrics):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]

        for (condition, (scratch_data, finetune_data)), color in zip(data_dict.items(), colors):
            epochs = range(1, len(scratch_data[metric]) + 1)
            ax.plot(epochs, scratch_data[metric], color=color, linestyle='--', label=f'no_paralogs_removed {condition}')
            ax.plot(epochs, finetune_data[metric], color=color, linestyle='-', label=f'paralogs_removed {condition}')
        
        ax.set_title(metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend(loc='best', fontsize='x-small', ncol=2)
        ax.set_ylim(0, 1)  # Assuming metrics are between 0 and 1

        # Set x-ticks to show fewer epoch numbers
        max_epoch = len(epochs)
        ax.set_xticks(np.linspace(1, max_epoch, 5, dtype=int))
        ax.tick_params(axis='x', rotation=45)

    # Remove any unused subplots
    for i in range(n_metrics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col] if n_rows > 1 else axes[col])

    plt.tight_layout()
    plt.savefig(f'viz/{specie}_all_conditions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    os.makedirs('viz', exist_ok=True)
    base_path = '/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results'
    conditions = ['80', '400', '2000', '10000']
    # species = ['arabidopsis', 'mouse', 'bee', 'zebrafish']
    species = ['MANE', 'mouse', 'honeybee', 'zebrafish']#, 'arabidopsis']
    
    metrics = ['acceptor_accuracy', 'acceptor_precision', 'acceptor_recall', 'acceptor_f1', 'acceptor_auprc',
               'donor_accuracy', 'donor_precision', 'donor_recall', 'donor_f1', 'donor_auprc',
               'accuracy']
    for specie in species:
        data_dict = {}
        combined_df = pd.DataFrame()

        for condition in conditions:
            no_paralog_removed = f'{base_path}/train_no_paralog_removal_outdir/{specie}/flanking_{condition}/SpliceAI_{specie}_train_{condition}_0_rs22/0/LOG/TEST'

            paralog_removed = f'{base_path}/train_outdir/{specie}/flanking_{condition}/SpliceAI_{specie}_train_{condition}_0_rs22/0/LOG/TEST'
            
            no_paralog_removed_data = process_directory(no_paralog_removed)
            print("=========")
            paralog_removed_data = process_directory(paralog_removed)

            data_dict[condition] = (no_paralog_removed_data, paralog_removed_data)
            print(no_paralog_removed_data.keys())
            print(paralog_removed_data.keys())

            # Add data to combined DataFrame
            for metric in metrics:
                print(f'{no_paralog_removed_data[metric]}')
                print(f'{paralog_removed_data[metric]}')
                combined_df[f'{metric}_scratch_{condition}'] = no_paralog_removed_data[metric]
                combined_df[f'{metric}_finetune_{condition}'] = paralog_removed_data[metric]

        # Plot comparison for all conditions
        plot_comparison(data_dict, metrics, specie)

        # Save combined DataFrame
        combined_df.to_csv(f'viz/{specie}_all_conditions_comparison.csv', index=False)

    print("Comparison complete. Results saved in CSV files and PNG files for each species.")

if __name__ == "__main__":
    main()
