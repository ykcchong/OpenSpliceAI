for species in MANE honeybee arabidopsis zebrafish mouse; do
# for species in RefSeq_noncoding; do
# for species in SpliceAI27; do
# for species in MANE; do
    mkdir -p ./viz/
    python plot_metrics_combined.py --species ${species} --output-dir ./viz/ --random-seeds 22
done