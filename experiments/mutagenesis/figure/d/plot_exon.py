import os
import gffutils
import matplotlib.pyplot as plt
import pandas as pd

############
# FILES
############
base = '/ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/d/'

# Input BED paths for each tool
donor_bed_osa = f'{base}/predict/pytorch/SpliceAI_5000_10000/donor_predictions.bed'
acceptor_bed_osa = f'{base}/predict/pytorch/SpliceAI_5000_10000/acceptor_predictions.bed'
donor_bed_sa = f'{base}/predict/keras/SpliceAI_5000_10000/donor_predictions.bed'
acceptor_bed_sa = f'{base}/predict/keras/SpliceAI_5000_10000/acceptor_predictions.bed'

# Raw scores CSV paths for each tool
scores_file_osa = f'{base}/results/exp_1/pytorch_10000_cftr/scores.csv'  # update path as needed
scores_file_sa = f'{base}/results/exp_1/keras_10000_cftr/scores.csv'    # update path as needed

gff_file = f"{base}/data/cftr.gff"
output_base = f"{base}/results/figures"

gene_start = 117480024
threshold = 0.5

############ SPLICE SITE MARKERS ###########    

# Read BED files into pandas DataFrames
bed1_d = pd.read_csv(donor_bed_osa, sep="\t", header=None, 
                     names=["chrom", "start", "end", "name", "score", "strand"])
bed1_a = pd.read_csv(acceptor_bed_osa, sep="\t", header=None, 
                     names=["chrom", "start", "end", "name", "score", "strand"])
bed2_d = pd.read_csv(donor_bed_sa, sep="\t", header=None, 
                     names=["chrom", "start", "end", "name", "score", "strand"])
bed2_a = pd.read_csv(acceptor_bed_sa, sep="\t", header=None, 
                     names=["chrom", "start", "end", "name", "score", "strand"])

# Create a GFF database (if not already created)
db = gffutils.create_db(gff_file, dbfn=f'{base}/data/cftr.db', 
                        merge_strategy='merge', force=True, keep_order=True)

# Extract exons for a specific gene
gene_id = "gene-CFTR"  # Replace with your gene ID
gene = db[gene_id]
exons = [(feature.start, feature.end) for feature in db.children(gene, featuretype="exon")]

# Define gene structure
gene_start = gene.start
gene_end = gene.end

############ Plotting Function ############

def plot_gene_with_scores(bed_d, bed_a, scores_file, title, output_filename, donor_marker_style, acceptor_marker_style):
    """
    Plots gene structure with splice sites and raw scores beneath.
    
    Parameters:
    - bed_d, bed_a: DataFrames for donor and acceptor sites.
    - scores_file: Path to CSV file with raw scores.
    - title: Plot title.
    - output_filename: Path for saving the figure.
    - donor_marker_style, acceptor_marker_style: Styling dictionaries.
    """
    # Load raw scores
    data = pd.read_csv(scores_file)
    positions = data['Position'] + gene_start
    donor_scores = data['Donor_Ref']
    acceptor_scores = data['Acceptor_Ref']
    
    # Create a figure with two vertical subplots sharing the x-axis
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(30, 2), sharex=True,
                                            gridspec_kw={'height_ratios': [1, 1.2]})
    
    #### Top Plot: Gene structure and splice site markers ####
    ax_top.hlines(y=0, xmin=gene_start, xmax=gene_end, color='black', linewidth=2)
    
    for exon in exons:
        ax_top.add_patch(plt.Rectangle((exon[0], -0.1), exon[1] - exon[0], 0.2, color='black'))
        
    # Add ">" marker for forward strand exon
    mark_start = sorted(exons)[0][0]
    mark_end = sorted(exons)[-1][1]
    interval = (mark_end - mark_start) // 12
    for reg in range(mark_start + interval, mark_end - interval, interval):
        ax_top.plot(reg, 0, marker='4', markersize=12, color='black')
    
    for _, row in bed_d.iterrows():
        if row['score'] >= threshold:
            feature_center = (row["start"] + row["end"]) / 2
            ax_top.plot(feature_center, donor_marker_style['offset'], 
                        marker=donor_marker_style['marker'], color=donor_marker_style['color'], 
                        markersize=8, alpha=0.5)
    for _, row in bed_a.iterrows():
        if row['score'] >= threshold:
            feature_center = (row["start"] + row["end"]) / 2
            ax_top.plot(feature_center, -acceptor_marker_style['offset'], 
                        marker=acceptor_marker_style['marker'], color=acceptor_marker_style['color'], 
                        markersize=8, alpha=0.5)
    
    ax_top.set_yticks([])
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.spines['left'].set_visible(False)
    ax_top.set_title(title)
    
    #### Bottom Plot: Raw scores ####
    ax_bottom.plot(positions, donor_scores, label='Donor', color='blue', alpha=0.5)
    ax_bottom.plot(positions, acceptor_scores, label='Acceptor', color='red', alpha=0.5)
    ax_bottom.set_ylabel('Score')
    
    # Remove x-axis ticks on top plot to avoid redundancy
    ax_top.set_xticks([])
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

############ Generate Plots with Raw Scores ############

# Define marker styles
donor_style = {'marker': 'v', 'color': 'blue', 'offset': 0.25}
acceptor_style = {'marker': '^', 'color': 'red', 'offset': 0.25}

# Plot for OpenSpliceAI with raw scores
plot_gene_with_scores(
    bed1_d, bed1_a,
    scores_file=scores_file_osa,
    title="OpenSpliceAI",
    output_filename=f"{output_base}/gene_plot_ospliceai_with_scores.png",
    donor_marker_style=donor_style,
    acceptor_marker_style=acceptor_style
)

# Plot for SpliceAI with raw scores
plot_gene_with_scores(
    bed2_d, bed2_a,
    scores_file=scores_file_sa,
    title="SpliceAI",
    output_filename=f"{output_base}/gene_plot_spliceai_with_scores.png",
    donor_marker_style=donor_style,
    acceptor_marker_style=acceptor_style
)