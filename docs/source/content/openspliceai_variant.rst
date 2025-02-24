
|


.. _variant_subcommand:

variant
=======

The ``variant`` subcommand evaluates the impact of genomic variants (SNPs and small INDELs) on splice sites by comparing model predictions on the reference (wild-type) sequence to predictions on the altered (mutant) sequence. It annotates a VCF file with “delta” scores for four events—donor gain, donor loss, acceptor gain, and acceptor loss—along with the relative position of each event. These delta scores reflect how much the mutation modifies the model’s predicted splice site strength and location.

|

Overview
--------

Similar to SpliceAI’s variant annotation approach (Jaganathan et al., 2019), the ``variant`` subcommand:

- **Parses** a user-provided VCF file.
- **Retrieves** the reference genome context from a FASTA file.
- **Loads** a trained OpenSpliceAI model (PyTorch or Keras).
- **Predicts** splice site probabilities for both wild-type and mutated sequences.
- **Computes** the maximum change in donor or acceptor probability (delta scores) within a fixed window (default ±50 nt) around each variant.
- **Outputs** an annotated VCF, including delta scores and delta positions for each allele.

|

Input Files
-----------

1. **VCF File**

   A standard VCF file containing variants (SNPs, small insertions, and deletions).

2. **Reference Genome (FASTA)**

   A FASTA file used to extract wild-type sequences. The subcommand checks that the reference allele in the VCF matches the reference genome.

3. **Annotation File**

   A gene annotation file that defines the genomic regions to consider. Variants outside of annotated genes or too close to the chromosome ends (by default, within the flanking region) are skipped.

4. **Trained Model Checkpoint(s)**

   A directory or file containing one or more OpenSpliceAI model checkpoints (PyTorch ``.pt`` or Keras ``.h5``). The subcommand can average predictions across multiple models.

|

Output Files
------------

The main output is a **VCF file** with added OpenSpliceAI annotations for each variant that passes filtering. Each variant line in the annotated VCF contains new fields indicating:

- **Delta Scores**: donor_gain, donor_loss, acceptor_gain, acceptor_loss
- **Delta Positions**: relative positions (±50 by default) of these maximum changes

|

Delta Score Computation
-----------------------

The “delta” score measures how much a mutation changes splice site predictions within a fixed window around the variant (default ±50 nucleotides). For each variant, we compute reference predictions (:math:`d_{ref}`, :math:`a_{ref}`) and alternative predictions (:math:`d_{alt}`, :math:`a_{alt}`) for donor (:math:`d`) and acceptor (:math:`a`) channels:

.. math::
   \mathrm{DS}(\mathrm{Acceptor\,Gain}) = \max\bigl(a_{alt} - a_{ref}\bigr)
   :label: eq:12

.. math::
   \mathrm{DS}(\mathrm{Acceptor\,Loss}) = \max\bigl(a_{ref} - a_{alt}\bigr)
   :label: eq:13

.. math::
   \mathrm{DS}(\mathrm{Donor\,Gain}) = \max\bigl(d_{alt} - d_{ref}\bigr)
   :label: eq:14

.. math::
   \mathrm{DS}(\mathrm{Donor\,Loss}) = \max\bigl(d_{ref} - d_{alt}\bigr)
   :label: eq:15

where each maximum is taken over a window of 101 positions (±50) centered on the variant. The position of the maximum difference is recorded as the “delta position” (negative if upstream of the variant, positive if downstream).

|

Usage
-----

.. code-block:: text

   usage: openspliceai variant [-h] -R reference -A annotation [-I [input]] [-O [output]] [-D [distance]] [-M [mask]] [--model MODEL] [--flanking-size FLANKING_SIZE]
                              [--model-type {keras,pytorch}] [--precision PRECISION]

   optional arguments:
      -h, --help            show this help message and exit
      -R reference          path to the reference genome fasta file
      -A annotation         "grch37" (GENCODE V24lift37 canonical annotation file in package), "grch38" (GENCODE V24 canonical annotation file in package), or path to a
                              similar custom gene annotation file
      -I [input]            path to the input VCF file, defaults to standard in
      -O [output]           path to the output VCF file, defaults to standard out
      -D [distance]         maximum distance between the variant and gained/lost splice site, defaults to 50
      -M [mask]             mask scores representing annotated acceptor/donor gain and unannotated acceptor/donor loss, defaults to 0
      --model MODEL, -m MODEL
                              Path to a SpliceAI model file, or path to a directory of SpliceAI models, or "SpliceAI" for the default model
      --flanking-size FLANKING_SIZE, -f FLANKING_SIZE
                              Sum of flanking sequence lengths on each side of input (i.e. 40+40)
      --model-type {keras,pytorch}, -t {keras,pytorch}
                              Type of model file (keras or pytorch)
      --precision PRECISION, -p PRECISION
                              Number of decimal places to round the output scores

|

Examples
--------

Example: Annotating Human Variants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openspliceai variant \
      --vcf input_variants.vcf \
      --ref-fasta GRCh38.fa \
      --annotation-file grch38.txt \
      --model /path/to/pytorch_models/ \
      --model-type pytorch \
      --dist-var 50 \
      --flanking-size 400 \
      --output-vcf annotated_variants.vcf

This command:

1. **Loads** the reference genome from ``GRCh38.fa``.
2. **Reads** the gene annotation from ``grch38.txt``.
3. **Scans** the directory ``/path/to/pytorch_models/`` for PyTorch checkpoints, averaging predictions from all found models.
4. **Filters** out variants near chromosome edges or outside annotated genes.
5. **Computes** donor and acceptor delta scores within ±50 nucleotides of each variant.
6. **Writes** a new VCF (``annotated_variants.vcf``) with appended columns for the four delta scores and positions.

|

Processing Pipeline
-------------------

1. **VCF Parsing**
   - For each variant, the subcommand checks if it lies within an annotated gene region.  
   - Variants that are too close to the chromosome ends or have reference alleles mismatching the FASTA are skipped.

2. **Reference & Mutant Sequence Extraction**
   - A window of :math:`2 \times \text{dist_var} + 1` around the variant is extracted from the FASTA (e.g., 101 nt for ``--dist-var=50``).
   - For each alternative allele, the subcommand constructs a mutant sequence by substituting the variant base(s).

3. **Model Loading & Prediction**
   - The user may supply **PyTorch** (``.pt``) or **Keras** (``.h5``) model checkpoints.  
   - If a directory is provided, predictions from each model are averaged.  
   - Wild-type (reference) and mutant sequences are one-hot encoded and fed into the model(s) in evaluation mode.

4. **Delta Score Calculation**
   - The difference in donor/acceptor probabilities (:math:`d_{alt}-d_{ref}`, :math:`a_{alt}-a_{ref}`) is computed across the window.  
   - The maximum positive or negative differences yield the four delta scores: donor gain, donor loss, acceptor gain, and acceptor loss (Equations :eq:`eq:12`–:eq:`eq:15`).

5. **VCF Annotation**
   - The four delta scores and their positions (relative to the variant site) are appended to each variant’s INFO field.  
   - The annotated VCF includes fields for each event, e.g. ``DG=...;DL=...;AG=...;AL=...;DGpos=...;DLpos=...;AGpos=...;ALpos=...``.

|

Conclusion
----------

The ``variant`` subcommand enables fine-grained analysis of how single-nucleotide changes or small INDELs affect splice site usage. By leveraging the same convolutional architecture as in OpenSpliceAI’s main pipeline, it provides a convenient, post-hoc annotation step for variant effect prediction. For further details on model loading, advanced configuration, or multi-model ensembles, consult the official OpenSpliceAI documentation.


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