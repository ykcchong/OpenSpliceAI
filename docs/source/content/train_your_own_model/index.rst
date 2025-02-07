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



Steps & Commands to Train OpenSpliceAI Models
=============================================

OpenSpliceAI offers a streamlined pipeline for training splice site prediction models on different species. Whether you aim to build a model using the Human MANE annotations or train a model for mouse, our framework provides a modular, efficient, and reproducible process.

The training workflow typically involves:

- **Data Preprocessing:**  
  Use the ``create-data`` subcommand to convert your reference genome and gene annotation files into HDF5-formatted training and test datasets.

- **Model Training:**  
  Train your model from scratch using the ``train`` subcommand, which employs a deep residual CNN architecture with adaptive learning rate scheduling and early stopping for optimal performance.

- **(Optional) Model Calibration:**  
  Fine-tune the output probabilities using the ``calibrate`` subcommand, ensuring that the predicted splice site probabilities accurately reflect true likelihoods.

For specific examples, please refer to the following OpenSpliceAI examples:

.. admonition:: OpenSpliceAI Examples
    :class: note

    * :ref:`train_your_own_model_mane` — Train the Human (MANE) model.
    * :ref:`train_your_own_model_mouse` — Train a model using mouse data.

.. toctree::
    :hidden:

    train_human_mane
    train_mouse

These example pages provide detailed step-by-step instructions and commands for training models on different species. For further information, consult the individual pages.

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