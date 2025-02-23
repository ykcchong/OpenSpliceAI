
|

.. _quick-start_create_data:

Quick Start Guide: create-data
==============================

This page provides a concise guide for using OpenSpliceAI's ``create-data`` subcommand, which converts genomic sequences and annotations into HDF5-formatted training/testing datasets.

|

Before You Begin
----------------

- **Install OpenSpliceAI**: Ensure you have installed OpenSpliceAI and its dependencies as described in the :ref:`Installation` page.
- **Acquire Reference Files**: You need a reference genome in FASTA format and a corresponding annotation (GFF/GTF) file.

|

Super-Quick Start
-----------------

1. **Reference Genome (FASTA)**:
   - Example: ``GCF_000001405.40_GRCh38.p14_genomic.fna``

2. **Reference Annotation (GFF/GTF)**:
   - Example: ``MANE.GRCh38.v1.3.refseq_genomic.gff``

To create training and testing HDF5 files:

.. code-block:: bash

   openspliceai create-data \
      --genome-fasta GCF_000001405.40_GRCh38.p14_genomic.fna \
      --annotation-gff MANE.GRCh38.v1.3.refseq_genomic.gff \
      --output-dir train_test_dataset/

After this step, you should see two main files (``dataset_train.h5`` and ``dataset_test.h5``) in the specified output directory, along with intermediate files. These HDF5 files contain one-hot-encoded gene sequences and corresponding splice site labels.

|

Next Steps
----------

- **Model Training**: Proceed to the :ref:`train_subcommand` to train an OpenSpliceAI model using the generated datasets.
- **Further Configuration**: Explore command-line options such as ``--biotype`` or ``--chr-split`` for customized dataset creation.

|
|
|
|
|


.. image:: ../../_images/jhu-logo-dark.png
   :alt: My Logo
   :class: logo, header-image only-light
   :align: center

.. image:: ../../_images/jhu-logo-white.png
   :alt: My Logo
   :class: logo, header-image only-dark
   :align: center