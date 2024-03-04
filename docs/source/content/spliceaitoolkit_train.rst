
|


.. _installation:

:code:`train`
===============

.. _sys-reqs:

System requirements
-------------------

.. admonition:: Software dependency

   * python >= 3.8.0
   * numpy >= 1.22.0
   * gffutils >= 0.10.1

These dependencies will be automatically installed when you install SpliceAI-toolkit through pip or conda. The only exception is **miniprot**. Since miniprot is not on PyPi, you will need to install it manually. Please check out the `miniprot installation guide <https://github.com/lh3/miniprot?tab=readme-ov-file#install>`_ on `GitHub <https://github.com/lh3/miniprot>`_.

.. admonition:: Version warning
   :class: important

   If your numpy version is >= 1.25.0, then it requires Python version >= 3.9. 
   
   Check out the scientific python ecosystem coordination guideline `SPEC 0 <https://scientific-python.org/specs/spec-0000/>`_ — Minimum Supported Versions to configure the package version compatibility.

   
..       $ conda create -n myenv python=3.10

|


There are three ways that you can install SpliceAI-toolkit:

.. _install-through-pip:

Install through pip
-------------------------

SpliceAI-toolkit is on `PyPi 3.12 <https://pypi.org/project/spliceai-toolkit/>`_ now. Check out all the releases `here <https://pypi.org/manage/project/spliceai-toolkit/releases/>`_. Pip automatically resolves and installs any dependencies required by SpliceAI-toolkit.

.. code-block:: bash
   
   $ pip install spliceai-toolkit

|

.. _install-through-conda: 

Install through conda
-------------------------------

Installing SpliceAI-toolkit through conda is the easiest way to go:

.. code-block:: bash
   
   TBC

   $ conda install -c bioconda spliceai-toolkit

|

.. _install-from-source:

Install from source
-------------------------

You can also install SpliceAI-toolkit from source. Check out the latest version on `GitHub <https://github.com/Kuanhao-Chao/SpliceAI-toolkit>`_
!

.. code-block:: bash

   $ git clone https://github.com/Kuanhao-Chao/SpliceAI-toolkit

   $ python setup.py install

|

.. _check-SpliceAI-toolkit-installation:

Check SpliceAI-toolkit installation
-------------------------------------

Run the following command to make sure SpliceAI-toolkit is properly installed:

.. code-block:: bash
   
   $ spliceai-toolkit -h


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

      usage: spliceai-toolkit [-h] {create-data,train,predict,variant} ...
      spliceai-toolkit: error: the following arguments are required: command

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