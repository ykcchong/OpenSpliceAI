.. raw:: html

    <script type="text/javascript">

        let mutation_fuc = function(mutations) {
            var dark = document.body.dataset.theme == 'dark';

            if (document.body.dataset.theme == 'auto') {
                dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
            }
            
            document.getElementsByClassName('sidebar_ccb')[0].src = dark ? './_static/JHU_ccb-white.png' : "./_static/JHU_ccb-dark.png";
            document.getElementsByClassName('sidebar_wse')[0].src = dark ? './_static/JHU_wse-white.png' : "./_static/JHU_wse-dark.png";



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
        document.addEventListener("DOMContentLoaded", mutation_fuc);
        var observer = new MutationObserver(mutation_fuc)
        observer.observe(document.body, {attributes: true, attributeFilter: ['data-theme']});
        console.log(document.body);
    </script>
    <link rel="preload" href="./_images/jhu-logo-dark.png" as="image">
    <div id="main_entry"></div>

|

.. _main:

.. raw:: html

    <embed>
        <div class="sidebar-logo-container" style="padding-bottom:-10px">
            <img class="sidebar-logo only-light" src="_static/logo_black.png" alt="Light Logo">
            <img class="sidebar-logo only-dark" src="_static/logo_white.png" alt="Dark Logo">
        </div>
    </embed>

.. image:: https://img.shields.io/badge/License-GPLv3-yellow.svg
    :target: https://img.shields.io/badge/License-GPLv3-yellow.svg

.. image:: https://img.shields.io/badge/version-v.0.0.1-blue
    :target: https://img.shields.io/badge/version-v.0.0.1-blue

.. image:: https://static.pepy.tech/personalized-badge/lifton?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=PyPi%20downloads
    :target: https://pepy.tech/project/lifton

.. image:: https://img.shields.io/github/downloads/Kuanhao-Chao/lifton/total.svg?style=social&logo=github&label=Download
    :target: https://github.com/Kuanhao-Chao/OpenSpliceAI/releases

.. image:: https://img.shields.io/badge/platform-macOS_/Linux-green.svg
    :target: https://github.com/Kuanhao-Chao/OpenSpliceAI/releases

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/Kuanhao-Chao/lifton/blob/main/notebook/lifton_example.ipynb




.. .. image:: https://img.shields.io/github/downloads/Kuanhao-Chao/LiftOn/total.svg?style=social&logo=github&label=Download
..     :target: https://img.shields.io/github/downloads/Kuanhao-Chao/LiftOn/total.svg?style=social&logo=github&label=Download

.. .. image:: https://img.shields.io/badge/platform-macOS_/Linux-green.svg
..     :target: https://img.shields.io/badge/platform-macOS_/Linux-green.svg

.. .. image:: https://colab.research.google.com/assets/colab-badge.svg
..     :target: https://colab.research.google.com/github/Kuanhao-Chao/LiftOn/blob/main/notebook/LiftOn_example.ipynb

| 


OpenSpliceAI is a flexible framework designed for easy retraining of the SpliceAI model with new datasets. It comes with models pre-trained on various species, including humans (MANE database), mice, thale cress (Arabidopsis), honey bees, and zebrafish. Additionally, the OpenSpliceAI is capable of processing genetic variants in VCF format to predict their impact on splicing.

|

Why OpenSpliceAI❓
=======================

1. **Easy-to-retrain framework**: Transitioning from the outdated Python 2.7, along with older versions of TensorFlow and Keras, the OpenSpliceAI is built on Python 3.7 and leverages the powerful PyTorch library. This simplifies the retraining process significantly. Say goodbye to compatibility issues and hello to efficiency — retrain your models with just two simple commands.
2. **Retrained on new dataset**: SpliceAI is great, but OpenSpliceAI makes it even better! The newly pretrained SpliceAI-Human model is updated from GRCh37 to GRCh38 human genome and integrates the latest MANE (Matched Annotation from NCBI and EMBL-EBI) annotations, ensuring that research is supported by the most up-to-date and precise genomic data available.
3. **Retrained on various species**:  Concerned that the SpliceAI model does not generalize to your study species because you are not studying humans? No problem! The OpenSpliceAI is released with models pretrained on various species, including human MANE, mouse, thale cress, honey bee, and zebrafish.
4. **Predict the impact of genetic variants on splicing**: Similar to SpliceAI, the OpenSpliceAI can take genetic variants in VCF format and predict the impact of these variants on splicing with any of the pretrained models.

OpenSpliceAI is open-source, free, and combines the ease of Python with the power of PyTorch for accurate splicing predictions.

|

Who is it for❓
====================================

1. If you want to study splicing in humans, just use the newly pretrained human SpliceAI-MANE! Better annotation, better results!
2. If you want to do splicing research in other species, the OpenSpliceAI has you covered! It comes with models pretrained on various species! And you can easily train your own SpliceAI with your own genome & annotation data.
3. If you are interested in predicting the impact of genetic variants on splicing, OpenSpliceAI is the perfect tool for you!

|

What does OpenSpliceAI do❓
====================================

* The OpenSpliceAI :code:`create-data` command takes a genome and annotation file as input and generates a dataset for training and testing your SpliceAI model.

* The OpenSpliceAI :code:`train` command uses the created dataset to train your own SpliceAI model.

* To avoid retraining your SpliceAI model from the ground up, the OpenSpliceAI :code:`fine-tune` command allows for the fine-tuning of the pretrained human model using your own created dataset. It tailors the model to better generalize to your specific species.

* The OpenSpliceAI :code:`predict` command takes a random gene sequence and predicts the score of each position, determining whether it is a donor, acceptor, or neither.

* The OpenSpliceAI :code:`variant` command takes a VCF file and predicts the impact of genetic variants on splicing.


|


Cite us
==================================


.. raw:: html
    
    <p>Kuan-Hao Chao, Alan Mao, Anqi Liu, Mihaela Pertea, and Steven L. Salzberg. <i>"OpenSpliceAI"</i> <b>bioRxiv</b>.
    <a href="https://doi.org/10.1093/bioinformatics/btaa1016" target="_blank"> <svg xmlns="http://www.w3.org/2000/svg" aria-hidden="true" x="0px" y="0px" viewBox="0 0 100 100" width="15" height="15" class="icon outbound"><path fill="currentColor" d="M18.8,85.1h56l0,0c2.2,0,4-1.8,4-4v-32h-8v28h-48v-48h28v-8h-32l0,0c-2.2,0-4,1.8-4,4v56C14.8,83.3,16.6,85.1,18.8,85.1z"></path> <polygon fill="currentColor" points="45.7,48.7 51.3,54.3 77.2,28.5 77.2,37.2 85.2,37.2 85.2,14.9 62.8,14.9 62.8,22.9 71.5,22.9"></polygon></svg> </a> </p>

    <p>Kishore Jaganathan, Sofia Kyriazopoulou Panagiotopoulou, Jeremy F. McRae, Siavash Fazel Darbandi, David Knowles, Yang I. Li, Jack A. Kosmicki, Juan Arbelaez, Wenwu Cui, Grace B. Schwartz, Eric D. Chow, Efstathios Kanterakis, Hong Gao, Amirali Kia, Serafim Batzoglou, Stephan J. Sanders, and Kyle Kai-How Farh. <i>"Predicting splicing from primary sequence with deep learning"</i> <b>Cell</b>.
    <a href="https://doi.org/10.1016/j.cell.2018.12.015" target="_blank"> <svg xmlns="http://www.w3.org/2000/svg" aria-hidden="true" x="0px" y="0px" viewBox="0 0 100 100" width="15" height="15" class="icon outbound"><path fill="currentColor" d="M18.8,85.1h56l0,0c2.2,0,4-1.8,4-4v-32h-8v28h-48v-48h28v-8h-32l0,0c-2.2,0-4,1.8-4,4v56C14.8,83.3,16.6,85.1,18.8,85.1z"></path> <polygon fill="currentColor" points="45.7,48.7 51.3,54.3 77.2,28.5 77.2,37.2 85.2,37.2 85.2,14.9 62.8,14.9 62.8,22.9 71.5,22.9"></polygon></svg> </a> </p>


|

User support
============
Please go through the :ref:`documentation <table-of-contents>` below first. If you have questions about using the package, a bug report, or a feature request, please use the GitHub issue tracker here:

https://github.com/Kuanhao-Chao/openspliceai/issues

|

Key contributors
================

OpenSpliceAI was designed and developed by `Kuan-Hao Chao <https://khchao.com/>`_.  This documentation was written by `Kuan-Hao Chao <https://khchao.com/>`_. The LiftOn logo was designed by `Kuan-Hao Chao <https://khchao.com/>`_.

|

.. _table-of-contents:

Table of contents
==================

.. toctree::
    :maxdepth: 2
    
    content/installation
    content/quickstart    
    
.. toctree::
    :caption: Subcommands usage
    :maxdepth: 2

    content/spliceaitoolkit_create-data
    content/spliceaitoolkit_train
    content/spliceaitoolkit_fine-tune
    content/spliceaitoolkit_predict
    content/spliceaitoolkit_variant


.. toctree::
    :caption: Pretrained models
    :maxdepth: 2

    content/pretrained_models/index

    content/output_explanation
    content/behind_scenes
    content/how_to_page
    content/function_manual

    content/changelog
    content/license
    content/contact

    


|
|
|
|
|

.. content/installation
..    content/quickstart
..    content/liftover_GRCh38_2_T2TCHM13
..    content/liftover_bee_insect
..    content/liftover_arabidopsis_plant
..    content/liftover_drosophila_erecta
..    content/liftover_mouse_2_rat
..    content/behind_scenes
..    content/how_to_page
..    content/function_manual
..    content/license
..    content/contact


.. image:: ./_images/jhu-logo-dark.png
   :alt: My Logo
   :class: logo, header-image only-light
   :align: center

.. image:: ./_images/jhu-logo-white.png
   :alt: My Logo
   :class: logo, header-image only-dark
   :align: center