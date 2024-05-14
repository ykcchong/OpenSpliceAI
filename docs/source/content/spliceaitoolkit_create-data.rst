
|


.. _installation:

:code:`create-data`
=====================

subcommand description
---------------------------------

This subcommand processes (1) **genomic sequences (FASTA)** and (2) **genome annotations (GFF/GTF)** into Hierarchical Data Format version 5 (HDF5). The input :math:`X` is one-hot-encoded pre-mRNA sequences, and the labels :math:`Y` for donor, acceptor , or non-splice sites are generated from genome annotations.

Functionality
+++++++++++++++++++++++++++++++++++
By default, the :code:`create-data` subcommand selects the longest transcript in a gene locus as the canonical transcript, mirroring the original SpliceAI canonical-transcript-labeling approach It processes only protein-coding genes, and it chunks each gene sequence every 5,000 nucleotides. For each gene, the processed dimension is :math:`L/5000 * (5000 + W) * 4`. Here, :math:`L` is the length of the gene sequence, :math:`W` is the flanking sequence size, and 4 is the number of nucleotides (A, C, G, T).

Configuration Options
+++++++++++++++++++++++++++++++++++
Chromosome Specification: Users can specify chromosomes for training and testing.
Automatic Splitting: If chromosomes are not specified, the toolkit automatically splits them, maintaining an approximate 80:20 ratio for training and testing datasets.

To process genomic sequences and genome annotations, ensure the input files are in the correct format (FASTA for sequences and GFF/GTF for annotations) and configure the options as needed. The output will be in HDF5 format, ready for downstream analysis.

|
|


Example of human MANE
---------------------------------


Input files
+++++++++++++++++++++++++++++++++++

To run this example, you will need to download the following two input files:

* **Input**

  1. reference **Genome** :math:`G` in FASTA : `GCF_000001405.40_GRCh38.p14_genomic.fna <ftp://ftp.ccb.jhu.edu/pub/data/spliceai-toolkit/train_data/spliceai-mane/GCF_000001405.40_GRCh38.p14_genomic.fna>`_
  2. reference **Annotation** :math:`A` in GFF3 : `MANE.GRCh38.v1.3.refseq_genomic.gff <ftp://ftp.ccb.jhu.edu/pub/data/spliceai-toolkit/train_data/spliceai-mane/MANE.GRCh38.v1.3.refseq_genomic.gff>`_


The command of spliceAI-toolkit to create training and testing datasets is as follows:


.. code-block:: bash

   spliceai-toolkit create-data \
   --genome-fasta  GCF_000001405.40_GRCh38.p14_genomic.fna \
   --annotation-gff MANE.GRCh38.v1.3.refseq_genomic.gff \
   --output-dir train_test_dataset_MANE/

After successfully running the :code:`create-data` subcommand, you will get the following two main files for model training and testing and other intermediate files:

* **Output**: 
  
  1. `dataset_train.h5 <ftp://ftp.ccb.jhu.edu/pub/data/spliceai-toolkit/train_data/spliceai-mane/dataset_train.h5>`_: this is the main file for model training. 
  2. `dataset_test.h5 <ftp://ftp.ccb.jhu.edu/pub/data/spliceai-toolkit/train_data/spliceai-mane/dataset_test.h5>`_: this is the main file for model testing. 
  3. intermediate files: 
      * datafile_train.h5
      * datafile_test.h5
      * stats.txt
       

|
|


Usage
------

.. code-block:: text

   usage: spliceai-toolkit create-data [-h] --annotation-gff ANNOTATION_GFF --genome-fasta GENOME_FASTA --output-dir OUTPUT_DIR [--parse-type {maximum,all_isoforms}] [--biotype {protein-coding,non-coding}]
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
