"""
Definitions of `keras` models.
"""

from keras.layers import Concatenate, Dense
from keras.models import Model

from lstm_ee.keras.models.funcs import get_inputs, modify_layer

from lstm_ee.keras.models.models_lstm import (
    make_standard_lstm_branch,
    make_standard_postprocess_branch,
    make_stacked_lstm_branch
)

from slice_lid.consts import DEF_MASK

def model_lstm_standard(
    lstm_units           = 16,
    layers_pre           = [],
    layers_post          = [],
    n_resblocks          = 0,
    max_prongs           = None,
    reg                  = None,
    batchnorm            = True,
    dropout              = None,
    vars_input_slice     = None,
    vars_input_png3d     = None,
    target_pdg_iscc_list = None,
):
    """Create the default SliceLID network.

    Parameters
    ----------
    lstm_units : int
        Number of units that LSTM layer that processes 3D prongs will have.
        Default: 16.
    layers_pre : list of int
        List of Dense layer sizes that will be used to preprocess prong inputs.
    layers_post : list of int
        List of Dense layer sizes that will be used to postprocess LSTM
        outputs.
    n_resblocks : int or None, optional
        Number of the fully connected residual blocks to be added before the
        output layer. Default: None
    max_prongs : int or None, optional
        Limit on the number of prongs that will be used. Default: None.
    reg : keras.Regularizer or None, optional
        Regularization to use. Default: None
    batchnorm : bool or None, optional
        Whether to use Batch Normalization. Default: True.
    dropout : float or None
        If not None then Dropout layers with `dropout` value of dropout will
        be added to regularize activations.
    vars_input_slice : list of str or None
        List of slice level input variable names.
    vars_input_png3d : list of str or None
        List of 3D prong level input variable names.
    target_pdg_iscc_list : list of (int, bool)
        This parameters specifies target classes of the `slice_lid` network.
        The specification has the form
            [ (pdg_1, iscc_1), (pdg_2, iscc_2), ... (pdg_N, iscc_N) ]
        C.f. `slice_lid.args.Config`.

    Returns
    -------
    keras.Model
        Model that defines the network.

    See Also
    --------
    model_lstm_stack
    """

    # pylint: disable=dangerous-default-value
    inputs = get_inputs(vars_input_slice, vars_input_png3d, None, max_prongs)
    # pylint: disable=unbalanced-tuple-unpacking
    input_slice, input_png3d = inputs

    branch_png3d = make_standard_lstm_branch(
        'png3d', input_png3d, layers_pre, lstm_units, batchnorm, dropout, reg,
        mask_value = DEF_MASK
    )

    layer_merged = Concatenate()([ branch_png3d, input_slice ])
    layer_merged = modify_layer(layer_merged, 'layer_merged', batchnorm)
    layer_post   = make_standard_postprocess_branch(
        layer_merged, layers_post, batchnorm, dropout, reg, n_resblocks
    )

    output = Dense(
        len(target_pdg_iscc_list) + 1,
        activation = 'softmax',
        name       = 'target',
    )(layer_post)

    model = Model(inputs = inputs, outputs = [ output ])

    return model

def model_lstm_stack(
    lstm_spec            = [ (32, 'forward'), ],
    layers_pre           = [],
    layers_post          = [],
    n_resblocks          = 0,
    max_prongs           = None,
    reg                  = None,
    batchnorm            = True,
    dropout              = None,
    vars_input_slice     = None,
    vars_input_png3d     = None,
    target_pdg_iscc_list = None,
    lstm_kwargs          = None
):
    """Create the SliceLID network with stacked LSTM layers.

    Parameters
    ----------
    lstm_spec : list of (int, str)
        List of pairs that specify number of units and directions of LSTM
        layers that will be used to process 3D prongs.
    layers_pre : list of int
        List of Dense layer sizes that will be used to preprocess prong inputs.
    layers_post : list of int
        List of Dense layer sizes that will be used to postprocess LSTM
        outputs.
    n_resblocks : int or None, optional
        Number of the fully connected residual blocks to be added before the
        output layer. Default: None
    max_prongs : int or None, optional
        Limit on the number of prongs that will be used. Default: None.
    reg : keras.Regularizer or None, optional
        Regularization to use. Default: None
    batchnorm : bool or None, optional
        Whether to use Batch Normalization. Default: True.
    dropout : float or None
        If not None then Dropout layers with `dropout` value of dropout will
        be added to regularize activations.
    vars_input_slice : list of str or None
        List of slice level input variable names.
    vars_input_png3d : list of str or None
        List of 3D prong level input variable names.
    target_pdg_iscc_list : list of (int, bool)
        This parameters specifies target classes of the `slice_lid` network.
        The specification has the form
            [ (pdg_1, iscc_1), (pdg_2, iscc_2), ... (pdg_N, iscc_N) ]
        C.f. `slice_lid.args.Config`.
    lstm_kwargs : dict or None, optional
        Extra arguments that will be passed to the LSTM layer constructors.
        Default: None

    Returns
    -------
    keras.Model
        Model that defines the network.

    See Also
    --------
    model_lstm_standard
    """


    # pylint: disable=dangerous-default-value
    inputs = get_inputs(vars_input_slice, vars_input_png3d, None, max_prongs)
    # pylint: disable=unbalanced-tuple-unpacking
    input_slice, input_png3d = inputs

    branch_png3d = make_stacked_lstm_branch(
        'png3d', input_png3d, layers_pre, lstm_spec,
        batchnorm, dropout, reg, lstm_kwargs,
        mask_value = DEF_MASK
    )

    layer_merged = Concatenate()([ branch_png3d, input_slice ])
    layer_merged = modify_layer(layer_merged, 'layer_merged', batchnorm)
    layer_post   = make_standard_postprocess_branch(
        layer_merged, layers_post, batchnorm, dropout, reg, n_resblocks
    )

    output = Dense(
        len(target_pdg_iscc_list) + 1,
        activation = 'softmax',
        name       = 'target',
    )(layer_post)

    model = Model(inputs = inputs, outputs = [ output ])

    return model

