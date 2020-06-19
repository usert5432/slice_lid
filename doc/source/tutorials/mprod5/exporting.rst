Using Trained Networks with NOvASoft
====================================

In order to use the network with ``NOvASoft`` it first must be exported
to the protobuf format. Then you can use it to calculate PID scores at the
NOvA-*art* or ``CAFAna`` levels.

Converting keras Network into Protobuf Format
---------------------------------------------

In order to convert the trained network into the protobuf format `slice_lid`
comes with a script ``scripts/tf/export_model.py``. For the proper operation
requires ``tensorflow`` version 1 package to be available on your system. The
usage of this script is simple:

.. code-block:: bash

   python scripts/tf/export_model.py NETWORK_PATH

This script should produce a directory ``NETWORK_PATH/tf`` with two files:

1. ``model.pb`` -- `slice_lid` network saved in protobuf format.
   This network is optimized for evaluation.
2. ``config.json`` -- network configuration that includes names of input
   variables that it uses, and names of input/output graph nodes.

You should copy this directory ``NETWORK_PATH/tf`` to the machine where it
will be used and maybe rename it to something nice like ``my_awesome_network``.


Using Network at NOvA-*art* Level
---------------------------------

To evaluate PID scores at the NOvA-*art* level you can use the
``FillSliceLID_module.cc`` *art* producer that is located under
``TensorFlowEvaluator/SliceLID/art/producer/``.
To make it work with your network simply modify its *fcl* configuration file
``FillSliceLIDConfig.fcl`` and replace there values of ``modelFHC``,
``modelRHC`` to point to your network(-s), for example

::

    modelFHC : "my_awesome_network"


Using Network at CAFAna Level
-----------------------------

To evaluate `slice_lid` classifier at ``CAFAna`` level, first you would have to
include the header with relevant definitions in your ``CAFAna`` script:

.. code-block:: cpp

    #include "TensorFlowEvaluator/SliceLID/cafana/SliceLIDVar.h"

Then you will need to create a ``CAFAnaModel`` object that will load network's
graph and evaluate it:

.. code-block:: cpp

   auto model = SliceLID::initCAFAnaModel("my_awesome_network");

Finally, this ``CAFAnaModel`` can be used to construct ``CAFAna`` variables
that perform the actual classification:

.. code-block:: cpp

   Var ncid     = SliceLID::ncid(model);
   Var numuid   = SliceLID::numuid(model);
   Var nueid    = SliceLID::nueid(model);
   Var nutauid  = SliceLID::nutauid(model);
   Var cosmicid = SliceLID::cosmicid(model);

These variables will allow you to get the NC, NuMu, NuE, etc scores predicted
by the `slice_lid` network.

.. warning::
    C.f. ``CAFAnaModel`` warning at similar location from the LSTM EE tutorial
    "MiniProd5 LSTM EE Retraining".

