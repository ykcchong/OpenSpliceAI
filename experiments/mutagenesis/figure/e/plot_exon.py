import gffutils
import matplotlib.pyplot as plt
import pandas as pd

############
# FILES
############
base = '/ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/e/predict/SpliceAI_5000_10000'
threshold = 0.5

donor_bed_osa = f'{base}/results/predict_openspliceai/predict_tp53_mut/SpliceAI_5000_10000_thres{threshold}/donor_predictions.bed'
acceptor_bed_osa = f'{base}/results/predict_openspliceai/predict_tp53_mut/SpliceAI_5000_10000_thres{threshold}/acceptor_predictions.bed'

donor_bed_sa = f'{base}/results/predict_spliceai/tp_53_mut/SpliceAI_5000_10000_thres{threshold}/donor_predictions.bed'
acceptor_bed_sa = f'{base}/results/predict_spliceai/tp_53_mut/SpliceAI_5000_10000_thres{threshold}/acceptor_predictions.bed'

gff_file = f"{base}/data/MANE_GRCh38_v1.3_TP53.gff"
output_base = f"{base}/results/figures"

############ SPLICE SITE MARKERS ###########    

# Read BED files into pandas DataFrames
bed1_d = pd.read_csv(donor_bed_osa, sep="\t", header=None, names=["chrom", "start", "end", "name", "score", "strand"])
bed1_a = pd.read_csv(acceptor_bed_osa, sep="\t", header=None, names=["chrom", "start", "end", "name", "score", "strand"])
bed2_d = pd.read_csv(donor_bed_sa, sep="\t", header=None, names=["chrom", "start", "end", "name", "score", "strand"])
bed2_a = pd.read_csv(acceptor_bed_sa, sep="\t", header=None, names=["chrom", "start", "end", "name", "score", "strand"])

# Create a GFF database
db = gffutils.create_db(gff_file, dbfn=f'{base}/data/tp53.db', merge_strategy='merge', force=True, keep_order=True)

# Extract exons for a specific gene
gene_id = "gene-TP53"  # Replace with your gene ID
gene = db[gene_id]
exons = [(feature.start, feature.end) for feature in db.children(gene, featuretype="exon")]

# Define gene structure
gene_start = gene.start
gene_end = gene.end

############ Combined Plot ###########    

# Create a combined plot
fig, ax = plt.subplots(figsize=(20, 4))

# Plot the gene line
ax.hlines(y=0, xmin=gene_start, xmax=gene_end, color='black', linewidth=2, label='Gene')

# Plot exons as rectangles
for exon in exons:
    ax.add_patch(plt.Rectangle((exon[0], -0.2), exon[1] - exon[0], 0.4, color='black', label='Exon'))

# Plot splice site markers from BED file 1 (green and blue markers above the line)
marker_offset = 0.22
size = 6
alpha = 0.5
for _, row in bed1_d.iterrows():
    feature_center = (row["start"] + row["end"]) / 2
    ax.plot(feature_center, marker_offset, marker="v", color="green", markersize=size, alpha=alpha, label="Donor Site OSA" if _ == 0 else "")
for _, row in bed1_a.iterrows():
    feature_center = (row["start"] + row["end"]) / 2
    ax.plot(feature_center, marker_offset, marker="v", color="blue", markersize=size, alpha=alpha, label="Acceptor Site OSA" if _ == 0 else "")

# Plot splice site markers from BED file 2 (green and blue markers below the line)
for _, row in bed2_d.iterrows():
    feature_center = (row["start"] + row["end"]) / 2
    ax.plot(feature_center, -marker_offset, marker="^", color="green", markersize=size, alpha=alpha, label="Donor Site SA" if _ == 0 else "")
for _, row in bed2_a.iterrows():
    feature_center = (row["start"] + row["end"]) / 2
    ax.plot(feature_center, -marker_offset, marker="^", color="blue", markersize=size, alpha=alpha, label="Acceptor Site SA" if _ == 0 else "")

# Add labels, legend, and formatting
ax.set_xlabel("Genomic Position")
ax.set_yticks([-0.4, 0, 0.4])
ax.set_yticklabels(["SA Markers", "Gene Structure", "OSA Markers"])
ax.set_title(f"Combined Exon-Intron Structure and Splice Site Markers for {gene_id}")
# ax.legend(loc="upper right")
ax.grid(False)

# Save the plot
plt.tight_layout()
plt.savefig(f'{output_base}/combined_plot.png', dpi=300)
plt.show()