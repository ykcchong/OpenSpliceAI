import os
import matplotlib.pyplot as plt

def read_aln_stats(aln_stats_file):
    """
    Read alignment statistics from a file.

    Parameters:
        aln_stats_file (str): Path to the file containing alignment statistics.

    Returns:
        tuple: Two lists, the first with alignment identity percentages and the
               second with coverage percentages.
    """
    ids = []
    cov = []
    with open(aln_stats_file, "r") as file:
        lines = file.read().splitlines()
        for line in lines:
            parts = line.split('\t')
            # Multiply by 100 to get percentages
            ids.append(float(parts[1]) * 100)
            cov.append(float(parts[2]) * 100)
    return ids, cov

def main():
    # Create output directory
    os.makedirs("viz", exist_ok=True)

    # Thresholds for drawing dashed lines
    x_threshold = 80
    y_threshold = 80

    # Define species names and corresponding dataset names
    species_names = ["Human-MANE", "Honeybee", "Arabidopsis", "Zebrafish", "Mouse"]
    exp_names     = ["MANE", "honeybee", "arabidopsis", "zebrafish", "mouse"]
    target = "test"  # your target dataset

    # Number of species (i.e. subplots)
    num_species = len(exp_names)
    
    # Create subplots: one row with a subplot per species.
    fig, axs = plt.subplots(1, num_species, figsize=(4 * num_species, 4.5), sharex=True, sharey=True)
    
    # If only one subplot is created, wrap it in a list so that the loop works.
    if num_species == 1:
        axs = [axs]

    # Loop over each species and plot the corresponding data.
    for idx, species in enumerate(exp_names):
        aln_fn = f"/home/kchao10/data_ssalzbe1/khchao/data/REDO_train_test_dataset/train_test_dataset_{species}/removed_paralogs.txt"
        print("Processing file:", aln_fn)
        ids, cov = read_aln_stats(aln_fn)
        
        ax = axs[idx]
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        
        # Make each subplot square
        ax.set_aspect('equal', adjustable='box')
        
        # Draw filled regions first so that they appear behind the dots.
        ax.fill_betweenx([0, y_threshold], 0, 105, color='green', alpha=0.2, zorder=1)
        ax.fill_betweenx([y_threshold, 105], 0, x_threshold, color='green', alpha=0.2, zorder=1)
        ax.fill_betweenx([y_threshold, 105], x_threshold, 105, color='red', alpha=0.2, zorder=1)
        
        # Draw the scatter plot on top.
        ax.scatter(ids, cov, s=4, zorder=2)
        
        # Draw dashed threshold lines.
        ax.vlines(x_threshold, 0, 105, colors='r', linestyles='dashed', zorder=3)
        ax.hlines(y_threshold, 0, 105, colors='r', linestyles='dashed', zorder=3)

        # Set axis labels
        ax.set_xlabel(f"Query ({target}) Alignment Identity (%)")
        ax.set_ylabel(f"Query ({target}) Coverage (%)")
        
        # Set the subplot title to the species name.
        ax.set_title(species_names[idx])

    # Set a common overall title for the entire figure.
    fig.suptitle(f"Alignments between train and {target} datasets", fontsize=20)
    
    # Adjust layout to make room for the overall title and ensure labels are not cut off.
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.1, top=0.88, wspace=0.3)
    
    # Save the figure with subplots.
    plt.savefig("viz/removed_paralogs_subplots.png", dpi=300)
    plt.clf()

if __name__ == "__main__":
    main()
