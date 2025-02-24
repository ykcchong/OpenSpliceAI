# for model_type in MANE species; do
# for model_type in species MANE; do
for model_type in species; do
    for species in MANE honeybee arabidopsis zebrafish mouse; do
    # for species in mouse; do
    # for species in RefSeq_noncoding; do
    # for species in SpliceAI27; do
    # for experiment in my_split_paralog_removal my_split_no_paralog_removal spliceai_default_no_paralog_removal MANE_cleaned_test_set; do
    # for experiment in MANE_cleaned_test_set; do
    # for experiment in new_model_arch_spliceai_default_paralog_removed; do
    # for species in MANE; do 
        python plot_metrics_combined.py --species ${species} --output-dir ./${model_type}/viz_${species}/ --experiment ${species} --openspliceai-model-type ${model_type}
    done
done