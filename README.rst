slice_lid
=========
Package to train NOvA SliceLID classifier.

Overview
--------
The `slice_lid` package contains a collection routines and scripts to train and
evaluate SliceLID classifier for the NOvA experiment. This package is inspired
by the `SliceLID <original_>`_.


Installation
------------
`slice_lid` package architecture resemble `lstm_ee`_ package and it reuses
large amount of code from it. Make sure that `lstm_ee` package is installed
first.

The `slice_lid` is intended for developers. Therefore, it is recommended to
install the live version of the package, for example:

1. Git clone this repository:

.. code-block:: bash

   git clone https://github.com/usert5432/slice_lid

2. Add cloned repo to the ``PYTHONPATH`` environment variable.

.. code-block:: bash

   export PYTHONPATH="FULL_PATH_TO_CLONED_REPO:${PYTHONPATH}"

You may want to add the line above to your ``~/.bashrc``.

For the proper operation `slice_lid` requires several other python packages to
be available on your system, see `Requirements`_.

If you are running `slice_lid` for the first time it might be useful to run
its test suite to make sure that the package is not broken:

.. code-block:: bash

    python -m unittest tests.run_tests.suite

Requirements
------------

`slice_lid` is written in python v3 and depends on the `lstm_ee`_ package
for the proper operation. Make sure to install `lstm_ee` and its dependencies.
Additionally, `slice_lid` has the following optional dependencies:

* ``sklearn`` -- for making t-SNE embedding plots.

Documentation
-------------

`slice_lid` package comes with sphinx documentation. The prebuilt version
of it can be found `link <prebuilt_doc_>`_ (requires nova credentials).
If you would like to compile it manually then run the following command
in the ``doc`` subdirectory (requires ``sphinx``):

.. code-block:: bash

   make html

It will build all available documentation, which can be viewed with a web
browser by pointing it to the ``build/html/index.html`` file.

In addition to the sphinx documentation the `slice_lid` code is covered by a
numpy like docstrings. Please refer to the docstrings and the source code for
the details about inner `slice_lid` workings.

.. _prebuilt_doc: https://nova-docdb.fnal.gov/cgi-bin/private/ShowDocument?docid=46199
.. _original: https://github.com/andrew1102/SliceLID
.. _lstm_ee:  https://github.com/usert5432/lstm_ee


