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

.. _quick-start_transfer:

Quick Start Guide: ``transfer``
======================================

This guide walks you through the essential steps for using the ``transfer`` subcommand to fine-tune your own OpenSpliceAI model. By leveraging a pre-trained model (for example, a human-trained model), you can convert HDF5 datasets - generated via the ``create-data`` subcommand - into a tailored deep learning model for splice site prediction.

|

Before You Begin
----------------

- **Installation:**  
  Follow the instructions on the :ref:`Installation` page to install OpenSpliceAI along with all necessary dependencies.

- **Pre-trained Model:**  
  Obtain a pre-trained OpenSpliceAI model in ``.pt`` format (e.g., `model_10000nt_rs10.pt <https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/models/spliceai-mane/10000nt/model_10000nt_rs10.pt>`_ |download_icon|).

- **Dataset Preparation:**  
  Generate the training and testing datasets for your species of interest using the ``create-data`` subcommand. See the :ref:`quick-start_create_data` guide for details. You will need ``dataset_train.h5`` and  ``dataset_test.h5``.

.. |download_icon| raw:: html

   <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
   <i class="fa fa-download"></i>


|

One-liner Start
---------------

If you have already generated the necessary files with the ``create-data`` subcommand — or if you prefer to download them directly from GitHub—proceed with the steps below:

1. **Training Dataset:**  
   `dataset_train.h5 <https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/examples/create-data/results/dataset_train.h5>`_ |download_icon|

2. **Testing Dataset:**  
   `dataset_test.h5 <https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/examples/create-data/results/dataset_test.h5>`_ |download_icon|

3. **Pre-trained Model:**  
   Download a pre-trained model from the `OpenSpliceAI GitHub repository <https://github.com/Kuanhao-Chao/OpenSpliceAI>`_.  
   For example:  
   `model_10000nt_rs10.pt <https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/models/spliceai-mane/10000nt/model_10000nt_rs10.pt>`_ |download_icon|

Execute the following command to initiate transfer learning:

.. code-block:: bash

   openspliceai transfer \
      --train-dataset dataset_train.h5 \
      --test-dataset dataset_test.h5 \
      --pretrained-model model_10000nt_rs10.pt \
      --flanking-size 10000 \
      --unfreeze-all \
      --epochs 10 \
      --early-stopping \
      --project-name new_species_transfer \
      --output-dir ./transfer_out/

This command will:

- Load the pre-trained model (``model_10000nt_rs10.pt``).
- Unfreeze all layers (using ``--unfreeze-all``).
- Fine-tuning the model on your custom dataset for 10 epochs, saving logs and checkpoints in the ``transfer_out/`` directory.


.. admonition:: Note
   :class: important

   Please note that the model transfer-learned in this experiment is not optimized for splice site prediction, as it was fine-tuned only on a small subset of the data. This example is intended solely to demonstrate the transfer-learning process. For a fully optimized, pre-trained model, please refer to the :ref:`pretrained_models_home` guide.

|

Next Steps
----------

After completing transfer learning, consider the following actions:

- **Explore ``transfer`` Options:**  
  Review the :ref:`transfer_subcommand` documentation to discover additional customization options for your transfer-learning process.

- **Calibration (Optional):**  
  Enhance the reliability of your model’s probability outputs by following the guidelines in the :ref:`quick-start_calibrate` guide.

- **Prediction:**  
  To deploy your newly trained model for splice site prediction, see the :ref:`quick-start_predict` guide.

- **Advanced Options:**  
  Experiment with further training parameters (such as adjusting the number of epochs or the patience value) to optimize model performance.

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