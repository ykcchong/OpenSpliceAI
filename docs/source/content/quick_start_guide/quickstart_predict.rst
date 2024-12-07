
|

.. _quick-start_predict:

Quick Start Guide Predict 
=========================

This page provides a straightforward quick-start guide to using OpenSpliceAI to (1) predict splice sites from DNA sequences. If you haven't already, please follow the steps outlined on the :ref:`Installation` page to install and load OpenSpliceAI.


Before you get started, make sure you have already cloned the `LiftOn OpenSpliceAI repository <https://github.com/Kuanhao-Chao/OpenSpliceAI>`_. We provide an example in `test/lifton_chr22_example.sh <https://github.com/Kuanhao-Chao/LiftOn/tree/main/test/lifton_chr22_example.sh>`_.


|

.. _super-quick-start:

Super-Quick Start (one-liner)
+++++++++++++++++++++++++++++++++++

OpenSpliceAI predicts splice sites from DNA sequences with 10,000 nt flanking sequences. To run OpenSpliceAI, all you need are three files:

1. A reference genome (**Genome** :math:`T`, FASTA Format):  `chm13_chr22.fa <https://github.com/Kuanhao-Chao/LiftOn/tree/main/test/chm13_chr22.fa>`_
2. A pretrained OpenSpliceAI model (**Genome** :math:`R`, FASTA Format): `GRCh38_chr22.fa <https://github.com/Kuanhao-Chao/LiftOn/tree/main/test/GRCh38_chr22.fa>`_

Run the following commands:

.. code-block:: bash

    $ cd test

    $ openspliceai -g GRCh38_chr22.gff3 -o GRCh38_2_CHM13_lifton.gff3 -copies chm13_chr22.fa GRCh38_chr22.fa

After this step, you will obtain ... We provide further explanations of the output file hierarchy in the :ref:`output files section <output_files>`.


|

.. _google-colab:

Try OpenSpliceAI on Google Colab
+++++++++++++++++++++++++++++++++++

We created a reproducible and easy-to-run OpenSpliceAI example on Google Colab. It's a good starting point, so go ahead and check it out!

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/Kuanhao-Chao/LiftOn/blob/main/notebook/lifton_example.ipynb


|

Congratulations! You have successfully installed and run OpenSpliceAI. For more detailed analysis explaination and file format, please check:

.. seealso::
    
    * :ref:`same_species-section`

    * :ref:`close_species-section`

    * :ref:`distant_species-section`

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
   :align: center