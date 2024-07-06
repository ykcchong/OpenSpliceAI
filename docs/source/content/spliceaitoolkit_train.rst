
|


.. _train_subcommand:

:code:`train`
===============

Subcommand description
---------------------------------

This subcommand takes **the training and testing HDF5 files**, which are the outputs from the :ref:`create_data_subcommand` subcommand, and trains a deep learning model to predict splice sites.

Input
+++++++++++++++++++++++++++++++++++

The output consists of the training and testing HDF5 files containing the processed gene sequences and their corresponding labels.

1. **training HDF5 file** : the dataset is used for model training.
2. **testing HDF5 file** :  the dataset is held out for model testing.

Output
+++++++++++++++++++++++++++++++++++

The main output file is the trained model in `PT` file, storing the model weights and architecture. The training and testing logs are also saved in log files.


Processing Steps
+++++++++++++++++++++++++++++++++++

Following are some of the processing steps for the :code:`train` subcommand:
The SpliceAI-pytorch is trained using the following steps with hyperparameters:

1. **Model architecture**:
   - 

3. **Optimization**:
   - The model utilizes the AdamW optimizer with initial learning rate 1e-3, and the `ReduceLROnPlateau` adaptive learning rate scheduler, with :code:`mode='min'`, :code:`factor=0.5`, and :code:`patience=2`. 

4. **Dataset Split**:
   - The training dataset is split into 90% for training and 10% for testing.

5. **Training**:
   - The model is trained for 20 epochs.
   - An early stopping condition is applied: if the validation loss does not improve for 5 consecutive epochs, the training stops early.


|
|

Example of human MANE
---------------------------------


Input files
+++++++++++++++++++++++++++++++++++

To run this example, you will need the following two files. They can be either downloaded through the provided links or generated using the :ref:`create_data_subcommand` subcommand.

1. `dataset_train.h5 <ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/train_data/spliceai-mane/dataset_train.h5>`_: this is the dataset for model training. 
2. `dataset_test.h5 <ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/train_data/spliceai-mane/dataset_test.h5>`_: this is the dataset for model testing. 


Commands
+++++++++++++++++++++++++++++++++++

The command of OpenSpliceAI to train the spliceAI-Pytorch model is as follows:


.. code-block:: bash

   openspliceai train --flanking-size 10000 \
   --exp-num full_dataset \
   --train-dataset /ccb/cybertron/khchao/data/train_test_dataset_MANE_test/dataset_train.h5 \
   --test-dataset /ccb/cybertron/khchao/data/train_test_dataset_MANE_test/dataset_test.h5 \
   --output-dir /ccb/cybertron/khchao/OpenSpliceAI/results/model_train_outdir/ \
   --project-name human_MANE_adeptive_lr \
   --random-seed 22 \
   --model SpliceAI \
   --loss cross_entropy_loss -d

After successfully running the :code:`train` subcommand, you will get the following trained model and log files: 


Output files
+++++++++++++++++++++++++++++++++++

* `dataset_train.h5 <ftp://ftp.ccb.jhu.edu/pub/data/OpenSpliceAI/spliceai-mane/SpliceAI-MANE-10000nt.pt>`_: the trained SpliceAI-Pytorch model.


|
|




Usage
------

.. code-block:: text

   usage: openspliceai train [-h] [--disable-wandb] --output-dir OUTPUT_DIR [--project-name PROJECT_NAME] [--flanking-size FLANKING_SIZE]
                                 [--random-seed RANDOM_SEED] [--exp-num EXP_NUM] [--train-dataset TRAIN_DATASET] [--test-dataset TEST_DATASET]
                                 [--loss LOSS] [--model MODEL]

   options:
   -h, --help            show this help message and exit
   --disable-wandb, -d
   --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                           Output directory to save the data
   --project-name PROJECT_NAME, -s PROJECT_NAME
   --flanking-size FLANKING_SIZE, -f FLANKING_SIZE
   --random-seed RANDOM_SEED, -r RANDOM_SEED
   --exp-num EXP_NUM, -e EXP_NUM
   --train-dataset TRAIN_DATASET, -train TRAIN_DATASET
   --test-dataset TEST_DATASET, -test TEST_DATASET
   --loss LOSS, -l LOSS  The loss function to train SpliceAI model
   --model MODEL, -m MODEL


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