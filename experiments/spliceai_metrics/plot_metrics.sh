for species in bee arabadop zebrafish mouse; do
    for num in 80 400 2000 10000; do
        python plot_metrics.py --species ${species} --output-dir ./vis/ \
        --flanking-size $num
    done
done