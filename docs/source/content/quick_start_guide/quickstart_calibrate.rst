
|

.. _quick-start_calibrate:

Quick Start Guide: calibrate
============================

This page provides a brief guide for using OpenSpliceAI's ``calibrate`` subcommand to adjust a trained model's probability outputs via temperature scaling.

|

Before You Begin
----------------

- **Trained Model**: A PyTorch model checkpoint (e.g., ``model_best.pt``) from :ref:`train_subcommand` or :ref:`transfer_subcommand`.
- **Test/Validation Dataset**: An HDF5 file (e.g., ``dataset_test.h5``) containing sequences and labels not used in training.

|

Super-Quick Start
-----------------

1. **Model Checkpoint**: ``model_best.pt``
2. **Test Dataset**: ``dataset_test.h5``

Run:

.. code-block:: bash

   openspliceai calibrate \
      --pretrained-model model_best.pt \
      --test-dataset dataset_test.h5 \
      --flanking-size 400 \
      --output-dir ./calibration_results/

Key steps:
- Loads the model and introduces a temperature parameter :math:`T`.
- Optimizes :math:`T` to improve calibration (reducing misalignment between predicted probabilities and observed outcomes).
- Outputs a ``temperature.pt`` file and calibration plots (e.g., reliability curves) in ``calibration_results/``.

|

Next Steps
----------

- **Predict**: Use the newly calibrated model to generate more reliable probability estimates with :ref:`predict_subcommand`.
- **Advanced Options**: Adjust arguments like ``--temperature-file`` to load or save existing temperature parameters.

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