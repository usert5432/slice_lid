Training SliceLID Networks
==========================

Prerequisites
-------------

Before you can do the training you should install the `slice_lid` package.
Please refer to :ref:`intro:Installation` for the instructions on how to do
that.

In addition to that you should setup the directory structure that `slice_lid`
expects, c.f. :ref:`manuals/directory_structure:Directory Structure Setup`.

Training
--------

The scripts to train mprod5 networks can be found in the `slice_lid` package
under the directory ``scripts/train/mprod5/``. In that directory you will
find two scripts named ``train_fd_{fhc,rhc}.py`` -- for FHC and RHC trainings.

If you want to train say FHC network then simply run

.. code-block:: bash

   python train_fd_fhc.py

Note, however that this script expects to find a dataset by a path specified
in a training configuration (in the file ``train_fd_fhc.py``):

.. code-block:: python

    config =
    ...
        'dataset'         : 'mprod5/fd_fhc/dataset_slice_lid_fd_fhc.csv.xz',
    ...

which is itself located under the ``${SLICE_LID_DATADIR}`` directory. So, make
sure that you have moved the dataset obtained in the previous step to the
following location

::

    "${SLICE_LID_DATADIR}/mprod5/fd_fhc/dataset_slice_lid_fd_fhc.csv.xz"

It may take anywhere from 30 minutes to several hours for the training to
complete, depending on your machine. You can speed up training by using
multiprocessing data generation and/or data caches. Please refer to the
"Caches and Multiprocessing" section of the `lstm_ee` documentation for the
details about speed optimization.

Training Results
----------------

Once training is complete the trained model and related files will be stored
under a directory specified in the training configuration (in the file
``train_fd_fhc.py``):

.. code-block:: python

    config =
    ...
        'outdir'          : 'mprod5/final/fd_fhc',
    ...

which is itself located under the ``${SLICE_LID_OUTDIR}`` directory. In other
words, you can find the trained model (``model.h5``), along with its
configuration (``config.json``) and training history (``log.csv``) under:

::

    "${SLICE_LID_OUTDIR}/mprod5/final/fd_fhc/model_hash(HASHSTRING)"

where HASHSTRING is an **MD5** hash of the training configuration. In the
remainder of this tutorial I will be referring to this directory as
**NETWORK_PATH**.


