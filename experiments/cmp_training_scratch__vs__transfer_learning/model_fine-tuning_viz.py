import os
import pandas as pd
import matplotlib.pyplot as plt

def read_data(file_path):
    with open(file_path, 'r') as f:
        return [float(line.strip()) for line in f]

def process_directory(dir_path):
    data = {}
    for filename in os.listdir(dir_path):
        if filename.endswith('.txt'):
            if filename == 'donor_topk_all.txt' or filename == 'acceptor_topk_all.txt' or filename == 'loss_every_update.txt':
                continue
            print(f'Reading {filename}...')
            metric_name = os.path.splitext(filename)[0]
            file_path = os.path.join(dir_path, filename)
            data[metric_name] = read_data(file_path)
    return data

def plot_comparison(scratch_data, finetune_data, metrics, specie, condition):
    n_metrics = len(metrics)
    n_cols = 5
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    fig.suptitle(f'Comparison of Metrics: {specie.capitalize()} - Flanking {condition}', fontsize=16)

    for i, metric in enumerate(metrics):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]

        epochs = range(1, len(scratch_data[metric]) + 1)
        ax.plot(epochs, scratch_data[metric], label='From Scratch')
        ax.plot(epochs, finetune_data[metric], label='Fine-tuned')
        ax.set_title(metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()
        ax.set_ylim(0, 1)  # Assuming metrics are between 0 and 1

        # Set x-ticks to show all epoch numbers
        ax.set_xticks(epochs)
        # ax.set_xticklabels(epochs, rotation=45, ha='right')

    # Remove any unused subplots
    for i in range(n_metrics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col] if n_rows > 1 else axes[col])

    plt.tight_layout()
    plt.savefig(f'{specie}_{condition}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    base_path = '/home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results'
    conditions = ['80', '400', '2000', '10000']
    # species = ['arabadop', 'mouse', 'bee', 'zebrafish']
    species = ['arabadop']
    
    metrics = ['acceptor_accuracy', 'acceptor_precision', 'acceptor_recall', 'acceptor_f1', 'acceptor_auprc',
               'donor_accuracy', 'donor_precision', 'donor_recall', 'donor_f1', 'donor_auprc',
               'accuracy']

    for specie in species:
        for condition in conditions:
            scratch_path = f'{base_path}/train_outdir/{specie}/flanking_{condition}/SpliceAI_{specie}_train_{condition}_0_rs22/0/LOG/TEST'
            finetune_path = f'{base_path}/fine-tune_outdir/{specie}/flanking_{condition}/SpliceAI_human_{specie}_fine-tune_{condition}_0_rs22/0/LOG/TEST'
            
            scratch_data = process_directory(scratch_path)
            finetune_data = process_directory(finetune_path)

            plot_comparison(scratch_data, finetune_data, metrics, specie, condition)

            # Create and save comparison DataFrame
            comparison_df = pd.DataFrame()
            for metric in metrics:
                comparison_df[f'{metric}_scratch'] = scratch_data[metric]
                comparison_df[f'{metric}_finetune'] = finetune_data[metric]
            
            comparison_df.to_csv(f'{specie}_{condition}_comparison.csv', index=False)

    print("Comparison complete. Results saved in CSV files and PNG files for each scenario.")

if __name__ == "__main__":
    main()
