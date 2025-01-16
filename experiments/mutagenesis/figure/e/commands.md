# MYBPC3
openspliceai predict --model /ccb/cybertron2/smao10/openspliceai/models/spliceai-mane/10000nt/model_10000nt_rs14.pth --flanking 10000 --input-sequence ./experiments/mutagenesis/figure/e/data/mybpc3.fa --output-dir ./experiments/mutagenesis/figure/e/predict/pytorch/ --threshold 0.5

python /ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/e/spliceai.py -o ./experiments/mutagenesis/figure/e/predict/keras/ -f 10000 -i ./experiments/mutagenesis/figure/e/data/mybpc3.fa -t 0.5

# CFTR
openspliceai predict --model /ccb/cybertron2/smao10/openspliceai/models/spliceai-mane/10000nt/ --flanking 10000 --input-sequence ./experiments/mutagenesis/figure/e/data/cftr.fa --output-dir ./experiments/mutagenesis/figure/e/predict/pytorch/ --threshold 0.1

python /ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/e/spliceai.py -o ./experiments/mutagenesis/figure/e/predict/keras/ -f 10000 -i ./experiments/mutagenesis/figure/e/data/cftr.fa -t 0.1