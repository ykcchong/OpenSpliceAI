# for species in bee arabadop zebrafish mouse; do
# for species in RefSeq_noncoding; do
for species in MANE; do
    python plot_metrics_combined.py --species ${species} --output-dir ./vis/
done