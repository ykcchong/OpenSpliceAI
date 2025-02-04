import os, sys
import matplotlib.pyplot as plt

def read_aln_stats(aln_stats_file):
    """
    Read alignment statistics from a file.

    Parameters:
    aln_stats_file (str): Path to the file containing alignment statistics.

    Returns:
    dict: A dictionary of alignment statistics.
    """
    ids = []
    cov = []
    with open(aln_stats_file, "r") as file:
        # aln_stats = {line.split()[0]: int(line.split()[1]) for line in file}
        lines = file.read().splitlines()
        # Skip the header lines
        for line in lines:
            parts = line.split('\t')
            ids.append(float(parts[1]) * 100)
            cov.append(float(parts[2]) * 100)
    return ids, cov

def main():
    # data_type = sys.argv[1]
    # print("data_type: ", data_type)
    os.makedirs("viz", exist_ok=True)
    x_threshold = 80
    y_threshold = 80
    species_names = ["Human-MANE", "Honeybee", "Thale Cress", "Zebrafish", "Mouse"]
    exp_names = ["MANE", "honeybee", "arabidopsis", "zebrafish", "mouse"]
    for idx, species in enumerate(exp_names):
        for target in ["test"]:
            aln_fn = f"/home/kchao10/data_ssalzbe1/khchao/data/REDO_train_test_dataset/train_test_dataset_{species}/removed_paralogs.txt"
            print("aln_fn: ", aln_fn)
            ids, cov = read_aln_stats(aln_fn)
            # ids = ids*100
            # cov = cov*100
            print("ids: ", ids)
            print("cov: ", cov)
            # Create a square plot
            plt.figure(figsize=(6, 6))  # 6x6 inches for a square figure
            plt.scatter(ids, cov, s=10)
            # Set the x and y axis limits from 0 to 1
            plt.xlim(0, 105)
            plt.ylim(0, 105)
            plt.vlines(x_threshold, 0, 105, colors='r', linestyles='dashed')
            plt.hlines(y_threshold, 0, 105, colors='r', linestyles='dashed')
            plt.xlabel(f"Query ({target}) Alignment Identity (%)")
            plt.ylabel(f"Query ({target}) Coverage (%)")
            plt.title(f"Alignments between train and {target} datasets ({species_names[idx]})", fontsize=14)
            # Add a filled region for x > 80 and y > 10
            # plt.fill_betweenx([y_threshold, 105], 0, 105, color='red', alpha=0.2)
            plt.fill_betweenx([0, y_threshold], 0, 105, color='green', alpha=0.2)
            plt.fill_betweenx([y_threshold, 105], 0, x_threshold, color='green', alpha=0.2)
            plt.fill_betweenx([y_threshold, 105], x_threshold, 105, color='red', alpha=0.2)

            plt.savefig(f"viz/{species}_removed_paralogs.png")
            # plt.savefig(f"/home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_{species}/removed_paralogs.png")
            plt.clf()


if __name__ == "__main__":  
    main()