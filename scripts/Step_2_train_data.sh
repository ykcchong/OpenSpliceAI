########################################
# Train bee SpliceAI-model
########################################
spliceai-toolkit train --flanking-size 80 \
--exp-num dataset_h5py_version \
--training-target bee \
--train-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/scripts/bee/dataset_train.h5 \
--test-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/scripts/bee/dataset_test.h5 \
--project-root /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/ \
--project-name arabadop_h5py_dataset \
--output-dir ./bee/ \
--model SpliceAI \
> train_splan.log 2> train_splan_error.log

# ########################################
# # Train arabadopsis SpliceAI-model
# ########################################
# spliceai-toolkit train --flanking-size 80 \
# --exp-num dataset_h5py_version \
# --training-target arabadop \
# --train-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/scripts/arabadop/dataset_train.h5 \
# --test-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/scripts/arabadop/dataset_test.h5 \
# --project-root /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/ \
# --project-name arabadop_h5py_dataset \
# --output-dir ./arabadop/ \
# --model SpliceAI \
# > train_splan.log 2> train_splan_error.log

########################################
# Train RefSeq SpliceAI-model
########################################
# spliceai-toolkit train --flanking-size 80 \
# --exp-num small_dataset_h5py_version \
# --training-target RefSeq \
# --train-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/results/train_test_dataset_RefSeq/dataset_train.h5 \
# --test-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/results/train_test_dataset_RefSeq/dataset_test.h5 \
# --project-root /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/ \
# --project-name RefSeq_h5py_dataset \
# --output-dir ./RefSeq/ \
# --model SpliceAI \
# > train_splan.log 2> train_splan_error.log