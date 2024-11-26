
|


.. _create_data_subcommand:

:code:`create-data`
=====================

Subcommand description
---------------------------------

This subcommand processes **genomic sequences (FASTA)** and **genome annotations (GFF/GTF)** files into Hierarchical Data Format version 5 (HDF5) files. 

Input
+++++++++++++++++++++++++++++++++++

1. reference **Genome** :math:`G` in FASTA: this file is the reference genome.
2. reference **Annotation** :math:`A` in GFF3: this file includes the annotated gene features.

Output
+++++++++++++++++++++++++++++++++++

The output consists of the training and testing HDF5 files containing the processed gene sequences and their corresponding labels.

Processing Steps
+++++++++++++++++++++++++++++++++++

For each gene sequence in the genome, the following processing steps are performed:

1. **Gene Sequence Processing**:
   
   - Each gene sequence is transformed into a 3D tensor, denoted as :math:`X`.
   - The shape of :math:`X` is :math:`⌈L/W⌉ \times (F + W) \times 4`, where: 
  
     - :math:`L` is the length of the gene sequence.
     - :math:`W` is the chunking window size (default is 5000nt in SpliceAI).
     - :math:`F` is the flanking sequence size (80nt, 400nt, 2,000nt, 10,000nt).
     - 4 represents the number of nucleotides (A, C, G, T).
   - The last dimension of :math:`X` is appended with Ns to make the length of each gene sequence a multiple of :math:`W`.

2. **Label Generation**:

   - Labels, denoted as :math:`Y`, are generated from the genome annotations.
   - The shape of :math:`Y` is :math:`⌈L/W⌉ \times W \times 3`, where each site in the gene sequence is labeled as:
     
     - Donor site
     - Acceptor site
     - Non-splice site

Example
+++++++++++++++++++++++++++++++++++

For a gene sequence of length 12,000 with a chunking window size (:math:`W`) of 5000 and 10k flanking sequence (:math:`F=10,000`), the resulting 3D tensor :math:`X` would have a shape of :math:`⌈12000/5000⌉ \times (10000 + 5000) \times 4 = 3 \times 5000 \times 4`.

The corresponding label tensor :math:`Y` would have a shape of :math:`⌈12000/5000⌉ \times 5000 \times 3 = 3 \times 5000 \times 3`.


|
|


Example of human MANE
---------------------------------


Input files
+++++++++++++++++++++++++++++++++++

To run this example, you will need to download the following two input files:

1. reference **Genome** :math:`G` in FASTA : `GCF_000001405.40_GRCh38.p14_genomic.fna <ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/train_data/spliceai-mane/GCF_000001405.40_GRCh38.p14_genomic.fna>`_
2. reference **Annotation** :math:`A` in GFF3 : `MANE.GRCh38.v1.3.refseq_genomic.gff <ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/train_data/spliceai-mane/MANE.GRCh38.v1.3.refseq_genomic.gff>`_


Commands
+++++++++++++++++++++++++++++++++++

The command of OpenSpliceAI to create training and testing datasets is as follows:


.. code-block:: bash

   openspliceai create-data \
   --genome-fasta  GCF_000001405.40_GRCh38.p14_genomic.fna \
   --annotation-gff MANE.GRCh38.v1.3.refseq_genomic.gff \
   --output-dir train_test_dataset_MANE/

After successfully running the :code:`create-data` subcommand, you will get the following two main files for model training and testing and other intermediate files:

Output files
+++++++++++++++++++++++++++++++++++
  
1. `dataset_train.h5 <ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/train_data/spliceai-mane/dataset_train.h5>`_: this is the main file for model training. 
2. `dataset_test.h5 <ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/train_data/spliceai-mane/dataset_test.h5>`_: this is the main file for model testing. 
3. intermediate files:

   * datafile_train.h5
   * datafile_test.h5
   * stats.txt
       

|
|


Usage
------

.. code-block:: text

   usage: openspliceai create-data [-h] --annotation-gff ANNOTATION_GFF --genome-fasta GENOME_FASTA --output-dir OUTPUT_DIR [--parse-type {maximum,all_isoforms}] [--biotype {protein-coding,non-coding}]
                                       [--chr-split {train-test,test}]

   optional arguments:
   -h, --help            show this help message and exit
   --annotation-gff ANNOTATION_GFF
                           Path to the GFF file
   --genome-fasta GENOME_FASTA
                           Path to the FASTA file
   --output-dir OUTPUT_DIR
                           Output directory to save the data
   --parse-type {maximum,all_isoforms}
                           Type of transcript processing
   --biotype {protein-coding,non-coding}
                           Biotype of transcript processing
   --chr-split {train-test,test}
                           The chromosome splitting approach for training and testing


|
|
|
|
|


.. image:: ../_images/jhu-logo-dark.png
   :alt: My Logo
   :class: logo, header-image only-light
   :align: center

.. image:: ../_images/jhu-logo-white.png
   :alt: My Logo
   :class: logo, header-image only-dark
