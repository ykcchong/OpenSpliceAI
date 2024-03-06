# ########################################
# # Train zebra_fish SpliceAI-model
# ########################################
# spliceai-toolkit train --flanking-size 80 \
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
# spliceai-toolkit train --flanking-size 80 \
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
# --exp-num full_dataset_h5py_version \
# --training-target RefSeq_canonical \
# --train-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/scripts/RefSeq_canonical/dataset_train.h5 \
# --test-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/scripts/RefSeq_canonical/dataset_test.h5 \
# --project-root /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/ \
# --project-name RefSeq_canonical_h5py_dataset \
# --output-dir ./RefSeq_canonical/ \
# --model SpliceAI \
# > train_splan_canonical.log 2> train_splan_canonical_error.log

spliceai-toolkit train --flanking-size 80 \
--exp-num full_dataset_h5py_version \
--train-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/results/train_test_dataset_MANE/dataset_train_all.h5 \
--test-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/results/train_test_dataset_MANE/dataset_test_0.h5 \
--output-dir /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/results/model_train_outdir/ \
--project-name human_MANE_newlog \
--model SpliceAI \
--loss focal_loss -d \ 
> train_splan_MANE_newlog.log 2> train_splan_MANE_newlog_error.log

# --loss focal_loss \ 
