
|

.. _quick-start_transfer:

Quick Start Guide: transfer
===========================

This page offers a simplified guide for using OpenSpliceAI's ``transfer`` subcommand, which fine-tunes a pre-trained model (e.g., human-trained) to another species or dataset via transfer learning.

|

Before You Begin
----------------

- **Trained Model**: Obtain a fully trained OpenSpliceAI model checkpoint (``.pt``) or directory of checkpoints.
- **Training & Testing Datasets**: You need HDF5 files (e.g., ``dataset_train.h5`` and ``dataset_test.h5``) created by the :ref:`create_data_subcommand`.

|

Super-Quick Start
-----------------

Suppose you have:
1. **Pre-trained Model**: ``OpenSpliceAI-MANE-best.pt``
2. **Target Dataset**: ``my_species_train.h5`` and ``my_species_test.h5``

Use the following command:

.. code-block:: bash

   openspliceai transfer \
      --train-dataset my_species_train.h5 \
      --test-dataset my_species_test.h5 \
      --pretrained-model OpenSpliceAI-MANE-best.pt \
      --flanking-size 400 \
      --unfreeze 2 \
      --epochs 10 \
      --early-stopping \
      --project-name new_species_transfer \
      --output-dir ./transfer_out/

This command:
- Loads the pre-trained model (``OpenSpliceAI-MANE-best.pt``).
- Unfreezes the last 2 layers (``--unfreeze 2``) for fine-tuning.
- Trains on your custom dataset for 10 epochs, storing logs and checkpoints in ``transfer_out/``.

|

Next Steps
----------

- **Calibration**: Optionally calibrate your fine-tuned model with the :ref:`calibrate_subcommand`.
- **Prediction**: Run the :ref:`predict_subcommand` to test or generate splice site predictions on new FASTA sequences.

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