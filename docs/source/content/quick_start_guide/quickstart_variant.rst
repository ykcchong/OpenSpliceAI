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

.. _quick-start_variant:

Quick Start Guide: ``variant``
==============================

This page summarizes how to use OpenSpliceAI's ``variant`` subcommand to assess the impact of genomic variants on splice sites.

|

Before You Begin
----------------

- **VCF File**: A variant call format file containing SNPs or small INDELs.
- **Reference Genome (FASTA)**: Must match the reference used in the VCF.
- **Annotation File**: Gene annotations to filter variants by genomic region.
- **Trained Model**: One or more OpenSpliceAI model checkpoints (PyTorch or Keras).

|

Super-Quick Start
-----------------

1. **Variants**: ``input.vcf``
2. **Reference FASTA**: ``hg19.fa``
3. **Annotation File**: ``grch37.txt``
4. **Model**: Directory containing PyTorch checkpoints (e.g., ``/models/pytorch/``)

Run:

.. code-block:: bash

    openspliceai variant \
      -R data/hg19.fa \
      -A data/grch37.txt \
      -m models/spliceai-mane/400nt/ \
      -f 400 \
      -t pytorch \
      -I data/input.vcf \
      -O examples/variant/output.vcf

This command:
- **Loads** the VCF variants and checks them against the reference genome.
- **Predicts** donor/acceptor scores for both wild-type and mutant sequences within ±50 nt.
- **Outputs** an annotated VCF (``output.vcf``) with delta scores and positions for donor/acceptor gain or loss.

|

Next Steps
----------

- **Review**: Inspect the appended INFO fields in the VCF for delta scores and their positions.
- **Further Analysis**: Filter or rank variants by largest delta scores to prioritize functional splicing impacts.

|
|
|
|
|


.. image:: ../_images/jhu-logo-dark.png
   :alt: My Logo
   :class: logo, header-image only-light
   :align: center

.. image:: ../_images/jhu-logo-white.png
   :alt: My Logo
   :class: logo, header-image only-dark
   :align: center