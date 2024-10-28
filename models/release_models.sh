for species in MANE honeybee zebrafish mouse arabidopsis; do
# for species in arabidopsis; do
    # echo "SPECIES: $species"
    echo "${species,,}"

    for flanking_size in 80 400 2000 10000; do  
    # for flanking_size in 10000; do  
        dir_root=spliceai-"${species,,}"/${flanking_size}nt
        echo $dir_root
        mkdir -p $dir_root
        for repeat_idx in {0..4}; do
        # for repeat_idx in 0 2 6 7 8; do
            # echo $repeat_idx
            cp /home/kchao10/data_ssalzbe1/khchao/OpenSpliceAI/results/train_outdir/FINAL/${species}/flanking_${flanking_size}/SpliceAI_${species}_train_${flanking_size}_${repeat_idx}_rs1${repeat_idx}/${repeat_idx}/models/model_best.pt $dir_root/model_${flanking_size}nt_rs1${repeat_idx}.pt
        done
    done
done