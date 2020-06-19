Directory Structure Setup
=========================

The `slice_lid` package scripts assume that the user datasets are located under
the directory specified by the ``${SLICE_LID_DATADIR}`` environment variable.
Therefore, before you begin training you should setup this environment variable
like (maybe worth putting this into your ``~/.bashrc``):

.. code-block:: bash

   export SLICE_LID_DATADIR=PATH_TO_DIRECTORY_WITH_DATA

Similarly, `slice_lid` scripts will save the trained models and the associated
data under the directory specified by the ``${SLICE_LID_OUTDIR}`` environment
variable. Therefore, you should setup this directory as well:

.. code-block:: bash

   export SLICE_LID_OUTDIR=PATH_TO_DIRECTORY_WHERE_RESULTS_WILL_BE_SAVED


