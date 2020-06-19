Obtaining Training/Validation Data for MiniProd5 Training
=========================================================

There are two ways to obtain training and validation datasets for the
`slice_lid` retraining:

1. You can regenerate datasets manually from the mprod5 **caf** files.
2. Or you can reuse already generated datasets.

I will provide instructions for each option below.

1. Manual Data Generation
-------------------------

The scripts to manually generate dataset from the mprod5 **caf** files are
committed to the NOvA devsrepo. You can fetch them using the following command

.. code-block:: bash

   export DEVSREPO=svn+ssh://p-novaart@cdcvs.fnal.gov/cvs/projects/novaart-devs
   svn checkout "${DEVSREPO}/trunk/users/torbunov/slice_lid/scripts/mprod5"

Inside the fetched directory *mprod5* you will find four files named
``exporter_slice_lid_fd_{fhc,rhc}.C``. They can be used to generate
training and validation datasets for the FHC and RHC horn currents
respectively. These scripts are known to be working at the
``R19-10-30-final-prod4.b`` release of NOvaSoft, and may work in the later
releases.

Job Submission
^^^^^^^^^^^^^^

Let say you want to generate data for the FHC training. To do that you would
need to submit a grid job that runs the script ``exporter_slice_lid_fd_fhc.C``
in parallel. The job can be submitted to the grid via the following command:

.. code-block:: bash

   submit_cafana.py \
        --njobs 250 --print_jobsub --rel R19-10-30-final-prod4.b \
        --outdir OUTDIR \
        exporter_slice_lid_fd_fhc.C

where ``OUTDIR`` is a directory under **/pnfs** where job output files will be
stored. Once the grid job has completed you will find multiple csv files under
``OUTDIR`` with names ``dataset_slice_lid_fd_fhc_*_of_*.csv``. These
output files need to be merged together before they can be used for training.

Merging Job Output Files
^^^^^^^^^^^^^^^^^^^^^^^^

To merge the ``*.csv`` files together you can use the ``merge_csv.sh`` script
that comes with the `lstm_ee` package. This script can be located in the
``scripts/data`` subdirectory of the `lstm_ee`.

In order to use ``merge_csv.sh`` to merge job output files you can run the
following command:

.. code-block:: bash

   bash merge_csv.sh MERGED_FILE_NAME.csv.xz OUTDIR/dataset_*.csv

After ``merge_csv.sh`` has finished running you can use the resulting file
``MERGED_FILE_NAME.csv.xz`` for training `slice_lid` networks.


2. Retrieving Old Datasets
--------------------------

The old mprod5 datasets are stored under the SAM system. The SAM definition
that contains these datasets is ``dataset_slice_lid_mprod5``. There are two
datasets available:

1. ``dataset_slice_lid_mprod5_fd_fhc.csv.xz`` -- FD FHC
2. ``dataset_slice_lid_mprod5_fd_rhc.csv.xz`` -- FD RHC

To retrieve any of those datasets you can use ``ifdh_fetch`` command, e.g.

::

    ifdh_fetch dataset_slice_lid_mprod5_fd_fhc.csv.xz

