
|


.. _installation:

Installation
===============

.. _sys-reqs:

OpenSpliceAI package provides the essential frameworks and libraries needed to train your own custom OpenSpliceAI models and apply the pretrained / self-trained models to the task of your choice. 

The pretrained OpenSpliceAI models are separated from the package and can be downloaded from `OpenSpliceAI ftp site <ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/>`_.
   - `OpenSpliceAI-MANE  <ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/spliceai-mane/>`_
   - `OpenSpliceAI-Mouse  <ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/spliceai-mouse/>`_
   - `OpenSpliceAI-Honeybee  <ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/spliceai-honeybee/>`_
   - `OpenSpliceAI-Thale-Cress  <ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/spliceai-arabidopsis/>`_
   - `OpenSpliceAI-Zebrafish  <ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/_spliceai-zebrafish/>`_
   - `SpliceAI Keras implementation <https://github.com/Illumina/SpliceAI/>`_

The following are the system requirements for OpenSpliceAI. There are three ways that you can install OpenSpliceAI:

.. _install-through-pip:

Install through pip
-------------------------

OpenSpliceAI is on `PyPi <https://pypi.org/project/OpenSpliceAI/>`_ now. Check out all the releases `here <https://pypi.org/manage/project/OpenSpliceAI/releases/>`_. Pip automatically resolves and installs any dependencies required by OpenSpliceAI.

.. code-block:: bash
   
   $ pip install openspliceai

|

.. _install-through-conda: 

Install through conda
-------------------------------

Installing OpenSpliceAI through conda is the easiest way to go:

.. code-block:: bash
   
   TBC

   $ conda install -c bioconda openspliceai

|

.. _install-from-source:

Install from source
-------------------------

You can also install OpenSpliceAI from source. Check out the latest version on `GitHub <https://github.com/Kuanhao-Chao/OpenSpliceAI>`_
!

.. code-block:: bash

   $ git clone https://github.com/Kuanhao-Chao/OpenSpliceAI

   $ python setup.py install

|

.. _check-OpenSpliceAI-installation:

Check OpenSpliceAI installation
-------------------------------------

Run the following command to make sure OpenSpliceAI is properly installed:

.. code-block:: bash
   
   $ openspliceai -h


.. dropdown:: Terminal output
    :animate: fade-in-slide-down
    :title: bg-light font-weight-bolder
    :body: bg-light text-left

    .. code-block::


      ====================================================================
      Deep learning framework to train your own SpliceAI model
      ====================================================================


      ███████╗██████╗ ██╗     ██╗ ██████╗███████╗ █████╗ ██╗   ████████╗ ██████╗  ██████╗ ██╗     ██╗  ██╗██╗████████╗
      ██╔════╝██╔══██╗██║     ██║██╔════╝██╔════╝██╔══██╗██║   ╚══██╔══╝██╔═══██╗██╔═══██╗██║     ██║ ██╔╝██║╚══██╔══╝
      ███████╗██████╔╝██║     ██║██║     █████╗  ███████║██║█████╗██║   ██║   ██║██║   ██║██║     █████╔╝ ██║   ██║
      ╚════██║██╔═══╝ ██║     ██║██║     ██╔══╝  ██╔══██║██║╚════╝██║   ██║   ██║██║   ██║██║     ██╔═██╗ ██║   ██║
      ███████║██║     ███████╗██║╚██████╗███████╗██║  ██║██║      ██║   ╚██████╔╝╚██████╔╝███████╗██║  ██╗██║   ██║
      ╚══════╝╚═╝     ╚══════╝╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝╚═╝      ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝   ╚═╝

      0.0.1

      usage: openspliceai [-h] {create-data,train,predict,variant} ...
      openspliceai: error: the following arguments are required: command

|

.. _installation-complete:

Now, you are ready to go !
--------------------------
Please continue to the :ref:`Quick Start Guide`.



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