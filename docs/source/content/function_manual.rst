
|

User Manual 
=======================

OpenSpliceAI
---------------------------------

.. code-block:: text
   :class: no-wrap
   
      ====================================================================
      Deep learning framework to train your own SpliceAI model
      ====================================================================


      ██████╗ ██████╗ ███████╗███╗   ██╗███████╗██████╗ ██╗     ██╗ ██████╗███████╗ █████╗ ██╗
      ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██╔══██╗██║     ██║██╔════╝██╔════╝██╔══██╗██║
      ██║   ██║██████╔╝█████╗  ██╔██╗ ██║███████╗██████╔╝██║     ██║██║     █████╗  ███████║██║
      ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║╚════██║██╔═══╝ ██║     ██║██║     ██╔══╝  ██╔══██║██║
      ╚██████╔╝██║     ███████╗██║ ╚████║███████║██║     ███████╗██║╚██████╗███████╗██║  ██║██║
      ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝     ╚══════╝╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝╚═╝

      0.0.1

      usage: openspliceai [-h] {create-data,train,test,calibrate,transfer,predict,variant} ...

      OpenSpliceAI toolkit to help you retrain your own splice site predictor

      positional arguments:
      {create-data,train,test,calibrate,transfer,predict,variant}
                              Subcommands: create-data, train, test, calibrate, predict, transfer, variant
         create-data         Create dataset for your genome for SpliceAI model training
         train               Train the SpliceAI model
         test                Test the SpliceAI model
         calibrate           Calibrate the SpliceAI model
         transfer            transfer a pre-trained SpliceAI model on new data.
         predict             Predict splice sites in a given sequence using the SpliceAI model
         variant             Label genetic variations with their predicted effects on splicing.

      optional arguments:
      -h, --help            show this help message and exit


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