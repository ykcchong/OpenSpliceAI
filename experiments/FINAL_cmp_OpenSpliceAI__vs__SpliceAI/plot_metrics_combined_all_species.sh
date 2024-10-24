# species_list="MANE honeybee zebrafish mouse"
species_list="arabidopsis"
for model_type in MANE species; do
    output_dir="./${model_type}/viz_all_species/"
    python plot_metrics_combined_all_species.py --species ${species_list} --output-dir ${output_dir} --experiment combined --openspliceai-model-type ${model_type}
done
