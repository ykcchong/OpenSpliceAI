
|


.. _installation:

Installation
===============

.. _sys-reqs:

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