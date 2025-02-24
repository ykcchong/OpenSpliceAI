.. raw:: html

    <script type="text/javascript">

        let mutation_lvl_1_fuc = function(mutations) {
            var dark = document.body.dataset.theme == 'dark';

            if (document.body.dataset.theme == 'auto') {
                dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
            }
            
            document.getElementsByClassName('sidebar_ccb')[0].src = dark ? '../../_static/JHU_ccb-white.png' : "../../_static/JHU_ccb-dark.png";
            document.getElementsByClassName('sidebar_wse')[0].src = dark ? '../../_static/JHU_wse-white.png' : "../../_static/JHU_wse-dark.png";



            for (let i=0; i < document.getElementsByClassName('summary-title').length; i++) {
                console.log(">> document.getElementsByClassName('summary-title')[i]: ", document.getElementsByClassName('summary-title')[i]);

                if (dark) {
                    document.getElementsByClassName('summary-title')[i].classList = "summary-title card-header bg-dark font-weight-bolder";
                    document.getElementsByClassName('summary-content')[i].classList = "summary-content card-body bg-dark text-left docutils";
                } else {
                    document.getElementsByClassName('summary-title')[i].classList = "summary-title card-header bg-light font-weight-bolder";
                    document.getElementsByClassName('summary-content')[i].classList = "summary-content card-body bg-light text-left docutils";
                }
            }

        }
        document.addEventListener("DOMContentLoaded", mutation_lvl_1_fuc);
        var observer = new MutationObserver(mutation_lvl_1_fuc)
        observer.observe(document.body, {attributes: true, attributeFilter: ['data-theme']});
        console.log(document.body);
    </script>

|


.. |download_icon| raw:: html

   <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

   <i class="fa fa-download"></i>


.. _train_your_own_model_mane:

Train Your Own Model – Human (MANE)
===================================

This guide provides a step-by-step walkthrough for training an OpenSpliceAI model on the human MANE (Matched Annotation from NCBI and EMBL-EBI) dataset. By following these instructions, you will generate training and testing datasets from the MANE annotations, train a splice site prediction model using a 10,000-nt flanking sequence, and optionally calibrate the model to improve probability estimates.

|

Prerequisites
-------------

Before you begin, ensure that you have:

- Installed OpenSpliceAI and its dependencies (see the :ref:`Installation` page).

- Cloned the OpenSpliceAI repository from the `OpenSpliceAI repository <https://github.com/Kuanhao-Chao/OpenSpliceAI>`_.

- Downloaded the necessary input files:

  - **Reference Genome (FASTA):** e.g., `GCF_000001405.40_GRCh38.p14_genomic.fna <https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.fna.gz>`_ |download_icon|

  - **Annotation File (GFF):** e.g., `MANE.GRCh38.v1.3.refseq_genomic.gff <https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/release_1.3/MANE.GRCh38.v1.3.refseq_genomic.gff.gz>`_ |download_icon|

|

Step 1: Create Training and Test Datasets
------------------------------------------

Use the ``create-data`` subcommand to process the genome FASTA file and the MANE GFF file into HDF5-formatted datasets. This step performs one-hot encoding of gene sequences and labels splice sites based on the annotations.

Run the following command:

.. code-block:: bash

   openspliceai create-data \
      --genome-fasta GCF_000001405.40_GRCh38.p14_genomic.fna \
      --annotation-gff MANE.GRCh38.v1.3.refseq_genomic.gff \
      --output-dir train_test_dataset_MANE/ \
      --remove-paralogs \
      --min-identity 0.8 \
      --min-coverage 0.8 \
      --parse-type canonical \
      --write-fasta \
      --split-method human \
      --canonical-only

**Explanation of key options:**

- ``--remove-paralogs``: Filters out paralogous sequences to avoid data leakage.
- ``--min-identity 0.8 --min-coverage 0.8``: Sets thresholds for sequence similarity when removing paralogs.
- ``--parse-type canonical``: Selects the longest (canonical) transcript per gene.
- ``--write-fasta``: Outputs intermediate FASTA files for reference.
- ``--split-method human``: Uses a human-specific strategy for splitting chromosomes into training and test sets.
- ``--canonical-only``: Restricts labeling to conserved splice site motifs.

After running this command, the output directory will contain two main files:
- ``dataset_train.h5`` for training
- ``dataset_test.h5`` for testing

|

Step 2: Train the OpenSpliceAI-MANE Model
------------------------------------------

Once your datasets are prepared, use the ``train`` subcommand to train the model. In this example, we train a model with 10,000-nt flanking sequences.

Run the following command:

.. code-block:: bash

   openspliceai train \
      --flanking-size 10000 \
      --exp-num full_dataset \
      --train-dataset train_test_dataset_MANE/dataset_train.h5 \
      --test-dataset train_test_dataset_MANE/dataset_test.h5 \
      --output-dir model_train_outdir/ \
      --project-name OpenSpliceAI-MANE \
      --random-seed 1 \
      --model SpliceAI \
      --loss cross_entropy_loss

**Key Options Explained:**

- ``--flanking-size 10000``: Specifies a flanking region of 10,000 nt, which has been shown to improve prediction accuracy.
- ``--exp-num full_dataset``: A label for this training experiment.
- ``--random-seed 1``: Ensures reproducibility of the training process.
- ``--model SpliceAI``: Indicates the use of the SpliceAI architecture.
- ``--loss cross_entropy_loss``: Uses categorical cross-entropy as the loss function.
- Output files (such as model checkpoints and logs) will be saved in the specified output directory.

|

Step 3 (Optional): Calibrate the Model
---------------------------------------

Calibration adjusts the model’s output probabilities so that they more accurately reflect true likelihoods. This is optional but recommended for improved interpretability.

Run the calibration command:

.. code-block:: bash

   openspliceai calibrate \
      --flanking-size 10000 \
      --train-dataset train_test_dataset_MANE/dataset_train.h5 \
      --test-dataset train_test_dataset_MANE/dataset_test.h5 \
      --output-dir model_calibrate_outdir/ \
      --project-name OpenSpliceAI-MANE-calibrate \
      --random-seed 1 \
      --pretrained-model model_train_outdir/model_best.pt \
      --loss cross_entropy_loss

**Highlights:**

- This command loads the best model checkpoint (``model_best.pt``) from training.
- It optimizes a temperature parameter to calibrate the output probabilities.
- Calibrated outputs and diagnostic plots (e.g., reliability curves) are saved to the specified directory.

|

Step 4: Use Your Trained Model
------------------------------

After training (and optional calibration), your model is ready for use. You can:

- **Predict Splice Sites:** Use the ``predict`` subcommand to generate splice site predictions from new FASTA files.
- **Analyze Variants:** Use the ``variant`` subcommand to assess the impact of genetic variants on splicing.


|

Conclusion
----------

By following these steps, you have successfully trained your own OpenSpliceAI model using the Human MANE annotation. This model can now be applied to predict splice sites and analyze the effects of genomic variants, offering a powerful tool for investigating gene regulation in human genomics.


|
|
|
|
|

.. image:: ../../_images/jhu-logo-dark.png
   :alt: OpenSpliceAI Logo
   :class: logo, header-image only-light
   :align: center

.. image:: ../../_images/jhu-logo-white.png
   :alt: OpenSpliceAI Logo
   :class: logo, header-image only-dark
   :align: center
