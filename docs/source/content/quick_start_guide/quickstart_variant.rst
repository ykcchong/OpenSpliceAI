
|

.. _quick-start_variant:

Quick Start Guide: variant
==========================

This page summarizes how to use OpenSpliceAI's ``variant`` subcommand to assess the impact of genomic variants on splice sites.

|

Before You Begin
----------------

- **VCF File**: A variant call format file containing SNPs or small INDELs.
- **Reference Genome (FASTA)**: Must match the reference used in the VCF.
- **Annotation File**: Gene annotations to filter variants by genomic region.
- **Trained Model**: One or more OpenSpliceAI model checkpoints (PyTorch or Keras).

|

Super-Quick Start
-----------------

1. **Variants**: ``input_variants.vcf``
2. **Reference FASTA**: ``GRCh38.fa``
3. **Annotation File**: ``grch38.txt``
4. **Model**: Directory containing PyTorch checkpoints (e.g., ``/models/pytorch/``)

Run:

.. code-block:: bash

   openspliceai variant \
      --vcf input_variants.vcf \
      --ref-fasta GRCh38.fa \
      --annotation-file grch38.txt \
      --model /models/pytorch/ \
      --model-type pytorch \
      --dist-var 50 \
      --flanking-size 400 \
      --output-vcf annotated_variants.vcf

This command:
- **Loads** the VCF variants and checks them against the reference genome.
- **Predicts** donor/acceptor scores for both wild-type and mutant sequences within ±50 nt.
- **Outputs** an annotated VCF (``annotated_variants.vcf``) with delta scores and positions for donor/acceptor gain or loss.

|

Next Steps
----------

- **Review**: Inspect the appended INFO fields in the VCF for delta scores and their positions.
- **Further Analysis**: Filter or rank variants by largest delta scores to prioritize functional splicing impacts.

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