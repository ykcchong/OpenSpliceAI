import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_flanking(averages, metrics, metric_labels, subsets, outdir):
    os.makedirs(outdir, exist_ok=True)
    sns.set(style="whitegrid")
    flanking_sizes = [80, 400, 2000, 10000]
    averages['flanking_size'] = pd.Categorical(averages['flanking_size'], categories=flanking_sizes, ordered=True)
    
    for subset in subsets:
        subset_data = averages[averages['subset_size'] == subset]
        for metric, label in zip(metrics, metric_labels):
            plt.figure(figsize=(10, 6))
            sns.pointplot(data=subset_data, x='flanking_size', y=metric, hue='model_type', markers='o', linestyles='-')
            plt.title(f'[Variant] {label} vs. flanking_size (Subset size: {subset})')
            plt.xlabel('Flanking Size')
            plt.ylabel(label)
            plt.legend(title='Model Type')
            plt.grid(True)
            plt.savefig(os.path.join(outdir, f'{metric}_subset_{subset}.png'), dpi=300)
            plt.close()  # Close the plot to avoid overlapping of figures

def plot_subset(averages, metrics, metric_labels, outdir): # Plot across subset sizes
    # Plot across subset sizes
    os.makedirs(outdir, exist_ok=True)
    sns.set(style="whitegrid")
    flanking_sizes = [80, 400, 2000, 10000]
    
    palette = sns.color_palette("husl", len(flanking_sizes) * 2)  # Create a distinct color palette

    for metric, label in zip(metrics, metric_labels):
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=averages, 
            x='subset_size', 
            y=metric, 
            hue='flanking_size', 
            style='model_type', 
            markers=True, 
            palette=palette
        )
        plt.title(f'[Variant] {label} vs. Subset Size')
        plt.xlabel('Subset Size (# variants in Mills/1000G dataset)')
        plt.ylabel(label)
        plt.legend(title='Flanking Size / Model Type')
        plt.grid(True)
        plt.savefig(os.path.join(outdir, f'{metric}_subset_size.png'), dpi=300)
        plt.close()


def main():
    metrics = ["elapsed_time_sec", "growth_rate", "max_footprint_mb", "n_avg_mb", "n_cpu_percent_c", "n_gpu_peak_memory_mb"]
    metric_labels = ["Elapsed Time (s)", "Memory Growth Rate (%)", "Peak Memory Footprint (MB)", "Average Memory (MB)", "CPU C Percent", "GPU Peak Memory (MB)"]
    subsets = [10, 50, 100, 250, 500, 1000]
    averages_file = 'aggregated_results.csv'
    averages = pd.read_csv(averages_file)
    outdir1 = './figure/plots_FLANKING'
    outdir2 = './figure/plots_SUBSET'
    

    plot_flanking(averages, metrics, metric_labels, subsets, outdir1)
    plot_subset(averages, metrics, metric_labels, outdir2)

if __name__ == '__main__':
    main()
