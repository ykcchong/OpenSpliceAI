import matplotlib.pyplot as plt
import numpy as np

# Data for each species
species = ["Human-MANE", "Mouse", "Zebrafish", "Honeybee", "Thale cress"]
train_sizes = [13704, 18049, 25970, 7448, 21143]
initial_test_sizes = [5522, 4143, 6747, 2487, 6414]
paralogous_removed = [39, 160, 2157, 2, 145]

# Calculate adjusted test sizes after removing paralogous sequences
adjusted_test_sizes = [initial - removed for initial, removed in zip(initial_test_sizes, paralogous_removed)]
combined_test_sizes = [adjusted + removed for adjusted, removed in zip(adjusted_test_sizes, paralogous_removed)]

# Set up the positions and bar width
x = np.arange(len(species))
bar_width = 0.4

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 3.4))

# Create bars for training set size
bars_train = ax.bar(x - bar_width/2, train_sizes, width=bar_width, label='Training Set Size')

# Create stacked bars for combined test size
bars_test = ax.bar(x + bar_width/2, adjusted_test_sizes, width=bar_width, label='Adjusted Test Set Size')
bars_paralogous = ax.bar(x + bar_width/2, paralogous_removed, width=bar_width, bottom=adjusted_test_sizes, label='Paralogous Sequences Removed')

# Add labels, title, and legend
ax.set_xlabel("Species", fontsize=12)
ax.set_ylabel("Gene Sequence Count", fontsize=12)
ax.set_title("Training vs. Combined Test Set (Adjusted & Paralogous) Sizes", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(species)
ax.legend()

# Show values on top of each section of the bars
for bar, train_size in zip(bars_train, train_sizes):
    ax.text(bar.get_x() + bar.get_width() / 2, train_size / 2, str(train_size), ha='center', va='center', color='white', fontsize=10)

for bar_test, adj_size in zip(bars_test, adjusted_test_sizes):
    ax.text(bar_test.get_x() + bar_test.get_width() / 2, adj_size / 2, str(adj_size), ha='center', va='center', color='black', fontsize=10)

for bar_para, removed, adj_size in zip(bars_paralogous, paralogous_removed, adjusted_test_sizes):
    ax.text(bar_para.get_x() + bar_para.get_width() / 2, adj_size + removed / 2, str(removed), ha='center', va='center', color='black', fontsize=10)

# Display the plot
plt.tight_layout()
plt.savefig("test_set_sizes.png", dpi=300)
