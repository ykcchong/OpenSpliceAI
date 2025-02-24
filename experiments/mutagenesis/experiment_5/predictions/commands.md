# tp53
openspliceai predict --model /ccb/cybertron2/smao10/openspliceai/models/spliceai-mane/10000nt/model_10000nt_rs14.pth --flanking 10000 --input-sequence ./experiments/mutagenesis/figure/d/data/tp53_grch38.fa --output-dir ./experiments/mutagenesis/figure/d/results/predict_tp53/ --threshold 0.5

# chr17 anno
openspliceai predict --model /ccb/cybertron2/smao10/openspliceai/models/spliceai-mane/10000nt/model_10000nt_rs14.pth --flanking 10000 --input-sequence /ccb/cybertron2/smao10/openspliceai/data/toy/human/GCF_000001405.40_GRCh38.p14_genomic.fna --annotation-file /ccb/cybertron2/smao10/openspliceai/data/toy/human/MANE_chr17.gff --output-dir ./experiments/mutagenesis/figure/d/results/predict_chr17/ --threshold 0.5

# chr17 noanno
openspliceai predict --model /ccb/cybertron2/smao10/openspliceai/models/spliceai-mane/10000nt/model_10000nt_rs14.pth --flanking 10000 --input-sequence /ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/d/data/chr17_grch38.fa --output-dir ./experiments/mutagenesis/figure/d/results/predict_chr17_raw/ --threshold 0.5

# tp53 no flank
openspliceai predict --model /ccb/cybertron2/smao10/openspliceai/models/spliceai-mane/10000nt/model_10000nt_rs14.pth --flanking 10000 --input-sequence ./experiments/mutagenesis/figure/d/data/tp53_grch38_noflank.fa --output-dir ./experiments/mutagenesis/figure/d/results/predict_tp53_noflank/ --threshold 0.5
 
# tp53 small
openspliceai predict --model /ccb/cybertron2/smao10/openspliceai/models/spliceai-mane/10000nt/model_10000nt_rs14.pth --flanking 10000 --input-sequence ./experiments/mutagenesis/figure/d/data/tp53_grch38_small.fa --output-dir ./experiments/mutagenesis/figure/d/results/predict_tp53_small/ --threshold 0.1


# Final 
## tp53 orig
openspliceai predict --model /ccb/cybertron2/smao10/openspliceai/models/spliceai-mane/10000nt/model_10000nt_rs14.pth --flanking 10000 --input-sequence ./experiments/mutagenesis/figure/d/data/tp53_grch38.fa --output-dir ./experiments/mutagenesis/figure/d/results/predict_tp53/ --threshold 0.5
openspliceai predict --model /ccb/cybertron2/smao10/openspliceai/models/spliceai-mane/10000nt/model_10000nt_rs14.pth --flanking 10000 --input-sequence ./experiments/mutagenesis/figure/d/data/tp53_grch38.fa --output-dir ./experiments/mutagenesis/figure/d/results/predict_tp53/ --threshold 0.01

python /ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/d/spliceai.py -o /ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/d/results/predict_spliceai/tp_53/ -f 10000 -i /ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/d/data/tp53_grch38.fa -t 0.5
python /ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/d/spliceai.py -o /ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/d/results/predict_spliceai/tp_53/ -f 10000 -i /ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/d/data/tp53_grch38.fa -t 0.01

## tp53 mutated
openspliceai predict --model /ccb/cybertron2/smao10/openspliceai/models/spliceai-mane/10000nt/model_10000nt_rs14.pth --flanking 10000 --input-sequence ./experiments/mutagenesis/figure/d/data/tp53_grch38_mutated_10421_A.fa --output-dir ./experiments/mutagenesis/figure/d/results/predict_tp53_mut/ --threshold 0.5
openspliceai predict --model /ccb/cybertron2/smao10/openspliceai/models/spliceai-mane/10000nt/model_10000nt_rs14.pth --flanking 10000 --input-sequence ./experiments/mutagenesis/figure/d/data/tp53_grch38_mutated_10421_A.fa --output-dir ./experiments/mutagenesis/figure/d/results/predict_tp53_mut/ --threshold 0.01

python /ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/d/spliceai.py -o /ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/d/results/predict_spliceai/tp_53_mut/ -f 10000 -i /ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/d/data/tp53_grch38_mutated_10421_A.fa -t 0.5
python /ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/d/spliceai.py -o /ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/d/results/predict_spliceai/tp_53_mut/ -f 10000 -i /ccb/cybertron2/smao10/openspliceai/experiments/mutagenesis/figure/d/data/tp53_grch38_mutated_10421_A.fa -t 0.01


