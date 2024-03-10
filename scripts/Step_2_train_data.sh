# ########################################
# # Train zebra_fish SpliceAI-model
# ########################################
# spliceai-toolkit train --flanking-size 10000 \
# --exp-num dataset_h5py_version \
# --training-target zebra_fish \
# --train-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/scripts/zebra_fish/dataset_train.h5 \
# --test-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/scripts/zebra_fish/dataset_test.h5 \
# --project-root /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/ \
# --project-name zebra_fish_h5py_dataset \
# --output-dir ./zebra_fish/ \
# --model SpliceAI \
# > train_splan_zebra_fish.log 2> train_splan_zebra_fish_error.log

########################################
# Train bee SpliceAI-model
########################################
# spliceai-toolkit train --flanking-size 10000 \
# --exp-num dataset_h5py_version \
# --training-target bee \
# --train-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/scripts/bee/dataset_train.h5 \
# --test-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/scripts/bee/dataset_test.h5 \
# --project-root /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/ \
# --project-name arabadop_h5py_dataset \
# --output-dir ./bee/ \
# --model SpliceAI \
# > train_splan.log 2> train_splan_error.log

# ########################################
# # Train arabadopsis SpliceAI-model
# ########################################
# spliceai-toolkit train --flanking-size 10000 \
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
# spliceai-toolkit train --flanking-size 10000 \
# --exp-num full_dataset_h5py_version \
# --training-target RefSeq_canonical \
# --train-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/scripts/RefSeq_canonical/dataset_train.h5 \
# --test-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/scripts/RefSeq_canonical/dataset_test.h5 \
# --project-root /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/ \
# --project-name RefSeq_canonical_h5py_dataset \
# --output-dir ./RefSeq_canonical/ \
# --model SpliceAI \
# > train_splan_canonical.log 2> train_splan_canonical_error.log

spliceai-toolkit train --flanking-size 10000 \
--exp-num full_dataset \
--train-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE_test/dataset_train.h5 \
--test-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE_test/dataset_test.h5 \
--output-dir /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/results/model_train_outdir/ \
--project-name human_MANE_relabel \
--model SpliceAI \
--loss cross_entropy_loss > train_splan_MANE_relabel_10000.log 2> train_splan_MANE_relabel_error_10000.log

# --loss focal_loss \ 
