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



Quick Start Guide
=================

OpenSpliceAI offers three primary workflows:

1. **Predict**: Use pre-trained models to directly predict splice sites from DNA sequences.
2. **Train from Scratch**: Build your own model by creating datasets and training from scratch.
3. **Transfer Learning**: Adapt an existing (e.g., human-trained) model to a new species or dataset.

The following sections provide a concise, step-by-step guide for each workflow.

|

Usage 1 – Predict
-----------------

.. admonition:: Quick Start: Predict with Pre-trained Models
    :class: note

    - :ref:`quick-start_predict`
    - :ref:`quick-start_variant`

In this workflow, you use a pre-trained OpenSpliceAI model to generate splice site predictions from your FASTA (and optionally GFF) files. This is ideal for users who want to quickly obtain splice site annotations without the need to train a model.


|

Usage 2 – Train from Scratch
----------------------------

.. admonition:: Quick Start: Train Your Own Model
    :class: note

    - :ref:`quick-start_create_data`
    - :ref:`quick-start_train`
    - :ref:`quick-start_calibrate` (Optional)
    - :ref:`quick-start_predict`
    - :ref:`quick-start_variant`

This workflow guides you through creating datasets from genomic sequences and annotations, training a SpliceAI model from scratch, optionally calibrating the model, and finally running predictions and variant analyses. It is best suited for users who want to build a custom model tailored to their data.


|

Usage 3 – Transfer Learning
---------------------------

.. admonition:: Quick Start: Transfer Learning Across Species
    :class: note

    - :ref:`quick-start_create_data`
    - :ref:`quick-start_transfer`
    - :ref:`quick-start_calibrate` (Optional)
    - :ref:`quick-start_predict`
    - :ref:`quick-start_variant`

This workflow enables you to adapt a pre-trained model (such as a human-trained model) to your target species using transfer learning. It involves generating datasets, fine-tuning the model, and then performing predictions and variant annotation. This approach is recommended when working with species for which limited training data is available.


.. |

.. Additional Resources
.. --------------------
.. For further details on installation, advanced usage, and file format specifications, please refer to the respective sections in the OpenSpliceAI documentation.

.. toctree::
    :hidden:

    quickstart_create-data
    quickstart_train
    quickstart_transfer
    quickstart_calibrate
    quickstart_predict
    quickstart_variant


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