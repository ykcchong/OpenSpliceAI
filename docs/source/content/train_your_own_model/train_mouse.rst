.. _train_your_own_model_mouse:

Train Your Own Model – Mouse
============================

This guide provides a detailed, step-by-step walkthrough for training an OpenSpliceAI model on mouse (Mus musculus) data. Using species-specific genome and annotation files, you can generate training and testing datasets and then train a splice site prediction model tailored to mouse. The procedure is analogous to the Human MANE workflow, with adjustments to input files and parameters as needed.

|

Prerequisites
-------------

Before you begin, make sure you have:

- **Installed OpenSpliceAI:** Follow the instructions on the :ref:`Installation` page.
- **Cloned the Repository:** Clone the OpenSpliceAI repository from the `LiftOn OpenSpliceAI repository <https://github.com/Kuanhao-Chao/OpenSpliceAI>`_.
- **Downloaded Input Files for Mouse:**
  - **Reference Genome (FASTA):** For example, the mouse genome assembly (e.g., GRCm39) can be obtained as ``mouse_genome.fna``.
  - **Annotation File (GFF):** A mouse gene annotation file (e.g., ``mouse_annotation.gff``) from Ensembl or NCBI.

|

Step 1: Create Training and Test Datasets
------------------------------------------

Use the ``create-data`` subcommand to process the mouse reference genome and annotation file into HDF5-formatted datasets. This step extracts gene sequences, selects canonical transcripts (typically protein-coding), and performs one-hot encoding of the sequences and splice site labels.

Run the command below (update file names as necessary):

.. code-block:: bash

   openspliceai create-data \
      --genome-fasta mouse_genome.fna \
      --annotation-gff mouse_annotation.gff \
      --output-dir train_test_dataset_mouse/ \
      --remove-paralogs \
      --min-identity 0.8 \
      --min-coverage 0.8 \
      --parse-type canonical \
      --write-fasta \
      --split-method mouse \
      --canonical-only

**Key Options Explained:**

- **--remove-paralogs:** Filters out paralogous sequences to avoid data leakage.
- **--min-identity 0.8 & --min-coverage 0.8:** Set thresholds for filtering.
- **--parse-type canonical:** Chooses the longest (canonical) transcript per gene.
- **--split-method mouse:** Applies mouse-specific rules for splitting the data into training and testing sets.
- **--canonical-only:** Limits labeling to conserved splice site motifs.

After running this command, check the output directory for two main files:
- ``dataset_train.h5`` (training dataset)
- ``dataset_test.h5`` (testing dataset)

|

Step 2: Train the Mouse Model
-----------------------------

With the datasets ready, use the ``train`` subcommand to train your mouse splice site prediction model. You can choose an appropriate flanking sequence length depending on your application; for this example, we use a 10,000-nt flanking size, which has proven effective in previous studies.

Run the following command (adjust file paths and parameters as needed):

.. code-block:: bash

   openspliceai train \
      --flanking-size 10000 \
      --exp-num mouse_full_dataset \
      --train-dataset train_test_dataset_mouse/dataset_train.h5 \
      --test-dataset train_test_dataset_mouse/dataset_test.h5 \
      --output-dir model_train_outdir_mouse/ \
      --project-name OpenSpliceAI-Mouse \
      --random-seed 42 \
      --model SpliceAI \
      --loss cross_entropy_loss

**Explanation of Options:**

- **--flanking-size 10000:** Sets the flanking region to 10,000 nt.
- **--exp-num mouse_full_dataset:** A label for this experiment.
- **--random-seed 42:** Ensures reproducibility.
- **--model SpliceAI:** Uses the SpliceAI model architecture.
- **--loss cross_entropy_loss:** Specifies the loss function.
- The model checkpoints (e.g., ``model_best.pt``) and training logs will be saved in the specified output directory.

|

Step 3 (Optional): Calibrate the Mouse Model
---------------------------------------------

Calibration refines the model’s output probabilities so that they better reflect empirical splice site likelihoods. If desired, run the calibration step using the ``calibrate`` subcommand.

Execute:

.. code-block:: bash

   openspliceai calibrate \
      --flanking-size 10000 \
      --train-dataset train_test_dataset_mouse/dataset_train.h5 \
      --test-dataset train_test_dataset_mouse/dataset_test.h5 \
      --output-dir model_calibrate_outdir_mouse/ \
      --project-name OpenSpliceAI-Mouse-calibrate \
      --random-seed 42 \
      --pretrained-model model_train_outdir_mouse/model_best.pt \
      --loss cross_entropy_loss

This command:
- Loads the best model checkpoint from your mouse training run.
- Optimizes the temperature parameter for calibration.
- Saves calibration outputs (e.g., temperature parameter, reliability curves) in the designated directory.

|

Step 4: Deploy Your Trained Mouse Model
-----------------------------------------

After training (and optional calibration), your mouse model is ready for use. You can now:
- **Predict Splice Sites:** Use the ``predict`` subcommand to run inference on new mouse FASTA sequences.
- **Analyze Variant Effects:** Use the ``variant`` subcommand to assess how specific mutations affect splicing in the mouse genome.

|

Conclusion
----------

By following these steps, you have successfully trained an OpenSpliceAI model on mouse data using species-specific genome and annotation files. This model can now be used to predict splice sites and analyze the impact of genetic variants in Mus musculus, thereby extending the utility of OpenSpliceAI to non-human genomics.

For further details on advanced configurations and troubleshooting, please refer to the full OpenSpliceAI documentation.

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
