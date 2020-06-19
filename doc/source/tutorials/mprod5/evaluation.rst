Evaluation of Trained Networks
==============================

The `slice_lid` package comes with a number of scripts to evaluate trained
networks. You can find them by exploring ``scripts/eval`` and
``scripts/studies`` directories of the `slice_lid`. The primary way to evaluate
networks is to calculate predicted error matrix (also known as confusion
matrix). This can be done by running the ``scripts/eval/eval_model.py`` script:

.. code-block:: bash

   python scripts/eval/eval_model.py [-e EXT] NETWORK_PATH

where ``EXT`` is a plot extension (e.g. *pdf*, *png*) and ``NETWORK_PATH`` is a
directory where network is saved.


Example of Plotting Error Matrices
----------------------------------

Let us try to plot error matrices for the network trained in the previous part
:doc:`training`. To create *pdf* plots of the matrices you would need to run

.. code-block:: bash

   python scripts/eval/eval_model.py -e pdf NETWORK_PATH

where **NETWORK_PATH** is the path with saved network. When the script
completes its execution it will produce a number of files in the directory

::

    "NETWORK_PATH/evals/cweight(none)"

These files are:

1. ``err_mat.txt`` components of the error matrix.
2. ``plots/err_mat.pdf`` -- plot of the unnormalized error matrix.
3. ``plots/err_mat_normed_truth.pdf`` -- plot of the error matrix normalized
   by truth (also known as efficiency at NOvA)
4. ``plots/err_mat_normed_preds.pdf`` -- plot of the error matrix normalized
   by predictions (also known as purity at NOvA)


