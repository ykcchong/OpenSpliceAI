import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_flanking(averages, metrics, metric_labels, subsets, outdir):
    os.makedirs(outdir, exist_ok=True)
    sns.set(style="whitegrid", font_scale=1.5)  # Increased font scale for larger text
    flanking_sizes = [80, 400, 2000, 10000]
    averages['flanking_size'] = pd.Categorical(
        averages['flanking_size'], categories=flanking_sizes, ordered=True
    )
    
    for subset in subsets:
        subset_data = averages[averages['subset_size'] == subset]
        for metric, label in zip(metrics, metric_labels):
            plt.figure(figsize=(12, 6))  # Increased figure size to accommodate legend
            sns.pointplot(
                data=subset_data, 
                x='flanking_size', 
                y=metric, 
                hue='model_type', 
                markers='o', 
                linestyles='-'
            )
            plt.title(f'[Variant] {label} vs. Flanking Size (Subset size: {subset})', fontsize=20)
            plt.xlabel('Flanking Size', fontsize=16)
            plt.ylabel(label, fontsize=16)
            plt.grid(True)
            plt.legend(
                title='Model Type', 
                bbox_to_anchor=(1.05, 1),  # Move legend outside the plot
                loc='upper left', 
                borderaxespad=0.
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(outdir, f'{metric}_subset_{subset}.png'), 
                dpi=300, 
                bbox_inches='tight'  # Ensure the entire plot is saved
            )
            plt.close()  # Close the plot to free memory

def plot_subset(averages, metrics, metric_labels, outdir):
    os.makedirs(outdir, exist_ok=True)
    sns.set(style="whitegrid", font_scale=1.5)
    
    # Ensure flanking_sizes are sorted and set as ordered categorical
    flanking_sizes = sorted(averages['flanking_size'].unique())
    averages['flanking_size'] = pd.Categorical(
        averages['flanking_size'], categories=flanking_sizes, ordered=True
    )

    # Create a color palette that scales with increasing flanking_size
    palette = sns.color_palette("viridis_r", len(flanking_sizes))

    # Rename the columns for better legend labels
    averages.rename(columns={'flanking_size': 'Flanking Size', 'model_type': 'Model Type'}, inplace=True)
    averages['Model Type'].replace({'keras': 'Keras', 'pytorch': 'PyTorch'}, inplace=True)

    # Get the unique model types after renaming
    model_types = sorted(averages['Model Type'].unique())

    # Define dash patterns 
    dash_patterns = [
        (None, None),        # Solid line
        (5, 5),              # Dashed line
        (1, 1),              # Dotted line
        (3, 5, 1, 5),        # Dash-dot line
        (3, 5, 1, 5, 1, 5),  # Dash-dot-dot line
    ]
    reversed_dash_patterns = dash_patterns[:len(model_types)][::-1]
    dashes_styles = dict(zip(model_types, reversed_dash_patterns))

    for metric, label in zip(metrics, metric_labels):
        plt.figure(figsize=(16, 9.7))
        sns.lineplot(
            data=averages,
            x='subset_size',
            y=metric,
            hue='Flanking Size',
            hue_order=flanking_sizes,
            style='Model Type',
            markers=True,
            markersize=12,  # Increase marker size
            palette=palette,
            style_order=model_types,
            dashes=dashes_styles,
            linewidth=3  # Thicker line for better visibility
        )
        plt.title(f'[Variant] {label} vs. Input Size', fontsize=30, fontweight='bold')
        plt.xlabel('Input Size (# Variants in Mills/1000G)', fontsize=27)
        plt.ylabel(label, fontsize=27)           
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        
        plt.grid(True)
        plt.legend(
            title='Model Size & Type',
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
            fontsize=20,      # Larger legend text
            title_fontsize=23 # Larger legend title text
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(outdir, f'{metric}_subset_size.png'),
            dpi=300,
            # bbox_inches='tight'
        )
        plt.close()

def main():
    metrics = [
        "elapsed_time_sec", 
        "growth_rate", 
        "max_footprint_mb", 
        "n_avg_mb", 
        "n_cpu_percent_c", 
        "n_gpu_peak_memory_mb"
    ]
    metric_labels = [
        "Elapsed Time (s)", 
        "Memory Growth Rate (%)", 
        "Peak Memory Footprint (MB)", 
        "Average Memory (MB)", 
        "CPU C Percent", 
        "GPU Peak Memory (MB)"
    ]
    subsets = [10, 50, 100, 250, 500, 1000]
    averages_file = '../aggregated_results.csv'
    averages = pd.read_csv(averages_file)
    outdir1 = './plots_FLANKING'
    outdir2 = './plots_SUBSET'
    
    # plot_flanking(averages, metrics, metric_labels, subsets, outdir1)
    plot_subset(averages, metrics, metric_labels, outdir2)

if __name__ == '__main__':
    main()
