import json
import matplotlib.pyplot as plt
import numpy as np

# Load data from JSON files
with open("donor_motifs.json") as f:
    donor_motifs = json.load(f)
with open("acceptor_motifs.json") as f:
    acceptor_motifs = json.load(f)

species = list(donor_motifs.keys())

# Calculate canonical/non-canonical counts
donor_counts = {"canonical": [], "non_canonical": []}
acceptor_counts = {"canonical": [], "non_canonical": []}

for sp in species:
    # Donor calculations
    canonical_d = donor_motifs[sp].get("GT", 0)
    non_canonical_d = sum(v for k,v in donor_motifs[sp].items() if k != "GT")
    donor_counts["canonical"].append(canonical_d)
    donor_counts["non_canonical"].append(non_canonical_d)
    
    # Acceptor calculations
    canonical_a = acceptor_motifs[sp].get("AG", 0)
    non_canonical_a = sum(v for k,v in acceptor_motifs[sp].items() if k != "AG")
    acceptor_counts["canonical"].append(canonical_a)
    acceptor_counts["non_canonical"].append(non_canonical_a)

# [Keep the rest of the visualization code unchanged from your example]
# [Add the plotting code here...]

# Set up the positions and bar width
x = np.arange(len(species))
bar_width = 0.35

# Plot the bar chart
fig, ax = plt.subplots(figsize=(10, 5))

# Donor bars
donor_canonical_bars = ax.bar(x - bar_width/2, donor_counts["canonical"], width=bar_width, label='Canonical Donor (GT)')
donor_non_canonical_bars = ax.bar(x - bar_width/2, donor_counts["non_canonical"], width=bar_width, bottom=donor_counts["canonical"], label='Non-Canonical Donor')

# Acceptor bars
acceptor_canonical_bars = ax.bar(x + bar_width/2, acceptor_counts["canonical"], width=bar_width, label='Canonical Acceptor (AG)')
acceptor_non_canonical_bars = ax.bar(x + bar_width/2, acceptor_counts["non_canonical"], width=bar_width, bottom=acceptor_counts["canonical"], label='Non-Canonical Acceptor')

# Add labels, title, and legend
ax.set_xlabel("Species", fontsize=12)
ax.set_ylabel("Motif Count", fontsize=12)
ax.set_title("Canonical vs Non-Canonical Motif Counts for Donors and Acceptors Across Species", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(species)
ax.legend()

# Show the ratio values on top of each stacked bar
for i, (donor_can, donor_non_can, acceptor_can, acceptor_non_can) in enumerate(zip(donor_counts["canonical"], donor_counts["non_canonical"], acceptor_counts["canonical"], acceptor_counts["non_canonical"])):
    donor_ratio = donor_can / (donor_can + donor_non_can) if donor_can + donor_non_can > 0 else 0
    acceptor_ratio = acceptor_can / (acceptor_can + acceptor_non_can) if acceptor_can + acceptor_non_can > 0 else 0
    ax.text(i - bar_width/2, donor_can + donor_non_can + 2500, f"{donor_ratio:.2%}", ha='center', va='bottom', color="black", fontsize=10)
    ax.text(i + bar_width/2, acceptor_can + acceptor_non_can + 2500, f"{acceptor_ratio:.2%}", ha='center', va='bottom', color="black", fontsize=10)

# Save plot
plt.tight_layout()
plt.savefig("splice_motif_analysis.png", dpi=300)
plt.show()