
|

.. _quick-start_predict:

Quick Start Guide: Predict
==========================

This page provides a streamlined guide for using OpenSpliceAI to **predict splice sites** from DNA sequences. If you haven't done so already, please see the :ref:`Installation` page for details on installing and configuring OpenSpliceAI.

Before You Begin
----------------

- **Clone the Repository**: Make sure you have cloned the `LiftOn OpenSpliceAI repository <https://github.com/Kuanhao-Chao/OpenSpliceAI>`_.  
- **Check Example Scripts**: We provide an example script, `test/lifton_chr22_example.sh <https://github.com/Kuanhao-Chao/LiftOn/tree/main/test/lifton_chr22_example.sh>`_, which demonstrates a sample pipeline using OpenSpliceAI.

Super-Quick Start (One-Liner)
-----------------------------

OpenSpliceAI can predict splice sites from DNA sequences with a default **10,000 nt** flanking region. You need:

1. **A reference genome (FASTA)**  
   Example: `chm13_chr22.fa <https://github.com/Kuanhao-Chao/LiftOn/tree/main/test/chm13_chr22.fa>`_

2. **A pre-trained OpenSpliceAI model or reference (FASTA)**  
   Example: `GRCh38_chr22.fa <https://github.com/Kuanhao-Chao/LiftOn/tree/main/test/GRCh38_chr22.fa>`_

Run the following commands (adapt or replace filenames as needed):

.. code-block:: bash

    cd test

    openspliceai -g GRCh38_chr22.gff3 \
                 -o GRCh38_2_CHM13_lifton.gff3 \
                 -copies chm13_chr22.fa GRCh38_chr22.fa

This command will generate a GFF file (``GRCh38_2_CHM13_lifton.gff3``) that contains predictions and any relevant coordinate transformations. (In this example, the command references a "lift" process, but the principle is similar for direct splice site predictions.)

After the process completes, you will see new output files in the directory. Refer to the :ref:`output_files` section for details on interpreting the results.

Try OpenSpliceAI on Google Colab
--------------------------------

We have created a reproducible Google Colab notebook to demonstrate OpenSpliceAI in a user-friendly environment:

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/Kuanhao-Chao/LiftOn/blob/main/notebook/lifton_example.ipynb

Click the badge above to open the notebook and run OpenSpliceAI interactively.

Summary
-------

Congratulations! You have successfully run OpenSpliceAI to predict splice sites. For a deeper dive into analysis options and file formats, please check:

.. seealso::

   * :ref:`same_species-section`
   * :ref:`close_species-section`
   * :ref:`distant_species-section`

We hope this quick start guide helps you get up and running with OpenSpliceAI. Happy predicting!

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