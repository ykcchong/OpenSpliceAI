import matplotlib.pyplot as plt
import numpy as np

# Data for each species
species = ["Human-MANE", "Mouse", "Zebrafish", "Honeybee", "Thale cress"]

# Canonical and non-canonical donor and acceptor motif counts
# Canonical motifs: Donor - GT, Acceptor - AG
donor_motifs = {
    "Human-MANE": {"GT": 181226, "GC": 1380, "AT": 214, "GA": 8, "TT": 8, "GG": 7},
    "Mouse": {"GT": 186880, "GC": 1533, "AT": 238, "GG": 20, "CC": 15, "TG": 2, "GA": 21, "TA": 9, "CT": 5, "CA": 8, "TC": 14, "AC": 6, "TT": 16, "AG": 9, "AA": 15, "CG": 1},
    "Zebrafish": {"GT": 285506, "GC": 2635, "AT": 610, "GG": 170, "CA": 147, "TG": 171, "CT": 111, "TT": 100, "GA": 175, "TA": 117, "CC": 57, "AG": 162, "TC": 76, "CG": 37, "AC": 111, "NN": 80, "AA": 141, "GN": 10, "AN": 7, "CN": 4, "TN": 1},
    "Honeybee": {"GT": 63906, "GC": 507, "AG": 12, "GA": 29, "AT": 70, "TA": 42, "TT": 22, "AA": 46, "CT": 8, "GG": 6, "TC": 8, "AC": 7, "CA": 11, "CC": 2, "CG": 4, "TG": 14},
    "Thale cress": {"GT": 117499, "GC": 1214, "AT": 109, "TT": 6, "AA": 2, "TA": 2, "GG": 8, "CA": 1, "GA": 11, "CT": 5, "AC": 2, "TG": 4, "AG": 3, "CG": 1},
}

acceptor_motifs = {
    "Human-MANE": {"AG": 182618, "AC": 187, "AT": 14, "GG": 10, "AA": 9, "TG": 4, "CT": 1},
    "Mouse": {"AG": 188396, "AC": 205, "GC": 4, "CT": 10, "GA": 7, "GG": 31, "TG": 26, "CA": 8, "AA": 24, "TC": 8, "AT": 31, "CC": 15, "TT": 17, "CG": 3, "GT": 2, "TA": 4, "NN": 1, "NG": 1},
    "Zebrafish": {"AG": 287905, "AC": 551, "GT": 128, "TT": 125, "CA": 184, "TG": 278, "GC": 98, "GG": 142, "CC": 91, "AA": 183, "AT": 151, "CT": 108, "GA": 92, "TA": 102, "TC": 113, "NN": 102, "NC": 6, "CG": 51, "NA": 6, "NT": 5, "NG": 6, "CN": 2},
    "Honeybee": {"AG": 64388, "TG": 37, "CT": 4, "AC": 42, "AA": 51, "AT": 43, "CG": 16, "TC": 10, "GT": 9, "TA": 24, "GG": 11, "TT": 28, "GA": 14, "CA": 10, "GC": 4, "CC": 3},
    "Thale cress": {"AG": 118711, "AC": 114, "GA": 1, "TA": 2, "CT": 3, "AT": 18, "CG": 2, "AA": 5, "CC": 1, "GT": 2, "TG": 2, "GG": 2, "TT": 1, "TC": 1, "CA": 2},
}

# Calculate canonical and non-canonical counts for donors and acceptors
donor_counts = {"canonical": [], "non_canonical": []}
acceptor_counts = {"canonical": [], "non_canonical": []}

for sp in species:
    # Donor motifs
    canonical_donor = donor_motifs[sp].get("GT", 0)
    non_canonical_donor = sum(count for motif, count in donor_motifs[sp].items() if motif != "GT")
    donor_counts["canonical"].append(canonical_donor)
    donor_counts["non_canonical"].append(non_canonical_donor)
    
    # Acceptor motifs
    canonical_acceptor = acceptor_motifs[sp].get("AG", 0)
    non_canonical_acceptor = sum(count for motif, count in acceptor_motifs[sp].items() if motif != "AG")
    acceptor_counts["canonical"].append(canonical_acceptor)
    acceptor_counts["non_canonical"].append(non_canonical_acceptor)

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

# Display the plot
plt.tight_layout()
plt.savefig("donor_acceptor_motif_counts.png", dpi=300)