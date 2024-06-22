FLANKING_SIZE=80
mkdir /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/results/model_train_outdir/spliceai27_MANE_${FLANKING_SIZE}/

python train_model.py \
--output /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/results/model_train_outdir/ \
--project-name spliceai27_MANE_${FLANKING_SIZE} \
--flanking-size 80 \
--exp-num 0 \
--train-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE_clean/dataset_train.h5 \
--test-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE_clean/dataset_test.h5 \
--model-type SpliceAI -d \
> /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/results/model_train_outdir/spliceai27_MANE_${FLANKING_SIZE}/model_train.log 2> /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/results/model_train_outdir/spliceai27_MANE_${FLANKING_SIZE}/model_train_error.log