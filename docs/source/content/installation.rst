
|


.. _installation:

Installation
===============

.. _sys-reqs:

System requirements
-------------------

.. admonition:: Software dependency

   * python >= 3.8.0
   * numpy >= 1.22.0
   * gffutils >= 0.10.1

These dependencies will be automatically installed when you install OpenSpliceAI through pip or conda. The only exception is **miniprot**. Since miniprot is not on PyPi, you will need to install it manually. Please check out the `miniprot installation guide <https://github.com/lh3/miniprot?tab=readme-ov-file#install>`_ on `GitHub <https://github.com/lh3/miniprot>`_.

.. admonition:: Version warning
   :class: important

   If your numpy version is >= 1.25.0, then it requires Python version >= 3.9. 
   
   Check out the scientific python ecosystem coordination guideline `SPEC 0 <https://scientific-python.org/specs/spec-0000/>`_ — Minimum Supported Versions to configure the package version compatibility.

   
..       $ conda create -n myenv python=3.10

|


There are three ways that you can install OpenSpliceAI:

.. _install-through-pip:

Install through pip
-------------------------

OpenSpliceAI is on `PyPi 3.12 <https://pypi.org/project/OpenSpliceAI/>`_ now. Check out all the releases `here <https://pypi.org/manage/project/OpenSpliceAI/releases/>`_. Pip automatically resolves and installs any dependencies required by OpenSpliceAI.

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