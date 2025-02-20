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

.. _quick-start_create_data:

Quick Start Guide: ``create-data``
==================================

This page provides a concise guide for using OpenSpliceAI's ``create-data`` subcommand, which converts genomic sequences and annotations into HDF5-formatted training/testing datasets.

|

Before You Begin
----------------

- **Install OpenSpliceAI**: Ensure you have installed OpenSpliceAI and its dependencies as described in the :ref:`Installation` page.
- **Acquire Reference Files**: You need a reference genome in FASTA format and a corresponding annotation (GFF/GTF) file.

|

.. |download_icon| raw:: html

   <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

   <i class="fa fa-download"></i>


One-liner Start
-----------------

1. **Reference Genome (FASTA):**
   
   - Example: 
     
     `GCF_000001405.40_GRCh38.p14_genomic_10_sample.fna <https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/examples/data/human/GCF_000001405.40_GRCh38.p14_genomic_10_sample.fna>`_ |download_icon|

2. **Reference Annotation (GFF/GTF):**
   
   - Example: 
     
     `MANE.GRCh38.v1.3.refseq_genomic_10_sample.gff <https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/examples/data/human/MANE.GRCh38.v1.3.refseq_genomic_10_sample.gff>`_ |download_icon|

To create training and testing HDF5 files:

.. code-block:: python

   openspliceai create-data \
      --remove-paralogs \
      --min-identity 0.8 \
      --min-coverage 0.8 \
      --parse-type canonical \
      --split-method human\
      --canonical-only \
      --genome-fasta GCF_000001405.40_GRCh38.p14_genomic_10_sample.fna \
      --annotation-gff MANE.GRCh38.v1.3.refseq_genomic_10_sample.gff \
      --output-dir train_test_dataset/

After this step, you should see two main files (``dataset_train.h5`` and ``dataset_test.h5``) in the specified output directory, along with intermediate files. These HDF5 files contain one-hot-encoded gene sequences and corresponding splice site labels.

|

Next Steps
-----------------

- **Explore ``create-data`` Options:**  
  Dive into the :ref:`create-data_subcommand` documentation to learn how to customize your dataset creation process.

- **Further Customization:**  
  Experiment with additional command-line options, such as ``--biotype`` and ``--chr-split``, for even more tailored dataset creation.

- **Begin Model Training:**  
  Follow the :ref:`quick-start_train` guide to start training your OpenSpliceAI model using your generated datasets.


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