# for species in MANE honeybee arabidopsis zebrafish mouse; do
# for species in RefSeq_noncoding; do
# for species in SpliceAI27; do
# for experiment in my_split_paralog_removal my_split_no_paralog_removal spliceai_default_no_paralog_removal MANE_cleaned_test_set; do
# for experiment in MANE_cleaned_test_set; do
# for experiment in new_model_arch_spliceai_default_paralog_removed; do
for species in arabidopsis honeybee mouse zebrafish; do 
    for model_trained_on in MANE GENCODE; do
        python plot_metrics_combined.py --species ${species} --output-dir ./${model_trained_on}/viz_${species}/ --random-seeds 22 --experiment ${species} --model-trained-on ${model_trained_on}
    done
done