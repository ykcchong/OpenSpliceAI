python train_model.py \
--output /Users/chaokuan-hao/Documents/Projects/splan/experiments/SpliceAI_train_code/model_train_outdir/ \
--project-name subset_training_dataset \
--flanking-size 80 \
--exp-num 0 \
--training-target SpliceAI27 \
--train-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/results/train_test_dataset_SpliceAI27/dataset_train_all.h5 \
--test-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/results/train_test_dataset_SpliceAI27/dataset_test_0.h5 > model_train.log 2> model_train_error.log