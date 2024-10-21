spliceai_gencode='/home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/SpliceAI-Gencode/'
openspliceai='/home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/'
# Align training sequences
minimap2 -a ${spliceai_gencode}spliceai_train.fa ${openspliceai}train.fa > ${openspliceai}minimap2/train_alignment.sam

# Align testing sequences
minimap2 -a ${spliceai_gencode}spliceai_test.fa ${openspliceai}test.fa > ${openspliceai}minimap2/test_alignment.sam
