spliceai-toolkit train --flanking-size 80 \
--exp-num small_dataset_h5py_version \
--training-target MANE \
--train-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/results/train_test_dataset_RefSeq/dataset_train.h5 \
--test-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/results/train_test_dataset_RefSeq/dataset_test.h5 \
--project-root /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/ \
--project-name RefSeq_h5py_dataset \
--output-dir ./tmp/ \
--model SpliceAI -d \
> train_splan.log 2> train_splan_error.log