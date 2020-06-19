"""
A collection of functions to setup keras training.
"""

from lstm_ee.train.setup import get_regularizer
from slice_lid.keras.models import model_lstm_standard, model_lstm_stack

def select_model(args):
    """Get `keras` model for the `slice_lid` training.

    The `keras` model will be selected from the models defined in the
    `slice_lid.keras` submodule.

    The `args.model` parameter will be used to select the `keras` model.
    The `args.model` should be a dict of the form form:
        { 'name' : NAME, 'kwargs' : KWARGS_DICT }.
    where NAME specified the name of the model and KWARGS_DICT is a dictionary
    of model parameters.

    Check the source code for the list of available models.

    See Also
    --------
    slice_lid.args.Config
    """

    name   = args.model['name']
    kwargs = args.model.get('kwargs', {}) or {}

    kwargs = {
        'reg'                  : get_regularizer(args.regularizer),
        'max_prongs'           : args.max_prongs,
        'vars_input_slice'     : args.vars_input_slice,
        'vars_input_png3d'     : args.vars_input_png3d,
        'target_pdg_iscc_list' : args.target_pdg_iscc_list,
        **kwargs
    }

    if name == 'standard':
        return model_lstm_standard(**kwargs)

    if name == 'stack_lstm':
        return model_lstm_stack(**kwargs)

    raise ValueError("Unknown model name: %s" % (name))

