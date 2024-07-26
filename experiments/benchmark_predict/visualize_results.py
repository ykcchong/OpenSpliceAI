import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_results(averages, metrics, subsets, outdir):

    sns.set(style="whitegrid")
    flanking_sizes = [80, 400, 2000, 10000]
    averages['flanking_size'] = pd.Categorical(averages['flanking_size'], categories=flanking_sizes, ordered=True)
    
    for subset in subsets:
        subset_data = averages[averages['subset_size'] == subset]
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            sns.pointplot(data=subset_data, x='flanking_size', y=metric, hue='model_type', markers='o', linestyles='-')
            plt.title(f'{metric} vs. flanking_size (Subset size: {subset})')
            plt.xlabel('Flanking Size')
            plt.ylabel(metric)
            plt.legend(title='Model Type')
            plt.grid(True)
            plt.savefig(os.path.join(outdir, f'{metric}_subset_{subset}.png'))
            plt.close()  # Close the plot to avoid overlapping of figures

def main():
    metrics = ["elapsed_time_sec", "growth_rate", "max_footprint_mb", "n_avg_mb", "n_cpu_percent_c", "n_gpu_peak_memory_mb"]
    subsets = [100, 200, 500, 1000]
    averages_file = 'aggregated_results.csv'
    averages = pd.read_csv(averages_file)
    outdir = './plots'
    os.makedirs(outdir, exist_ok=True)

    plot_results(averages, metrics, subsets, outdir)

if __name__ == '__main__':
    main()
