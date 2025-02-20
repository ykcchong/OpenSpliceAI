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

.. _quick-start_calibrate:

Quick Start Guide: ``calibrate``
=================================

This guide provides a brief walkthrough for using the ``calibrate`` subcommand in OpenSpliceAI to adjust your trained model’s probability outputs via temperature scaling.

.. admonition:: Note
   :class: important

   Calibration is an optional step that can further enhance the reliability of model predictions. Our research demonstrates that OpenSpliceAI’s output probabilities generally reflect real-world likelihoods. Nevertheless, running this calibration step can serve as a useful double-check, generating reliability curves and score distribution plots for your review.

|


.. |download_icon| raw:: html

   <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

   <i class="fa fa-download"></i>


Before You Begin
----------------

Ensure you have the following prerequisites:

- **Pre-trained Model:**  
  Obtain a pre-trained OpenSpliceAI model in ``.pt`` format (for example, 
  `model_10000nt_rs10.pt <https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/models/spliceai-mane/10000nt/model_10000nt_rs10.pt>`_ |download_icon|) from either the :ref:`train_subcommand` or :ref:`transfer_subcommand`.

- **Test/Validation Dataset:**  
  Prepare an HDF5 file generated from a test set (e.g., ``dataset_test.h5``) containing sequences and labels that were not used during training.

|

Quick Start
-----------

1. **Testing Dataset:**  
   Download the test dataset:  
   `dataset_test.h5 <https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/examples/create-data/results/dataset_test.h5>`_ |download_icon|

2. **Pre-trained Model:**  
   Download the pre-trained model from the `OpenSpliceAI GitHub repository <https://github.com/Kuanhao-Chao/OpenSpliceAI>`_. For example:  
   `model_10000nt_rs10.pt <https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/models/spliceai-mane/10000nt/model_10000nt_rs10.pt>`_ |download_icon|

Run the following command to start the calibration process:

.. code-block:: bash

   openspliceai calibrate \
      --pretrained-model model_10000nt_rs10.pt \
      --test-dataset dataset_test.h5 \
      --flanking-size 10000 \
      --output-dir ./calibration_results/

|

Key Steps in Calibration
-------------------------

- **Model Loading:**  
  The pre-trained model is loaded and a temperature parameter (:math:`T`) is introduced.

- **Temperature Optimization:**  
  The parameter :math:`T` is optimized to better align the predicted probabilities with observed outcomes, thus improving calibration.

- **Output Generation:**  
  An optimized temperature parameter is saved to a ``temperature.pt`` file, and calibration plots (e.g., reliability curves) are generated in the ``calibration_results/`` directory.

|

Next Steps
----------

- **Explore Calibration Options:**  
  For more details on available arguments and further customization, refer to the :ref:`calibrate_subcommand` documentation.

- **Prediction:**  
  Apply your newly calibrated model to generate more reliable probability estimates by following the :ref:`predict_subcommand` guide.


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