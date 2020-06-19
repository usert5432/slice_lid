"""
Functions to train `slice_lid` models.
"""

import logging
import numpy as np

from lstm_ee.train.setup import (
    get_default_callbacks, get_optimizer, get_keras_concurrency_kwargs
)

from slice_lid.args.args   import Args
from slice_lid.data.data   import load_data
from slice_lid.train.setup import select_model

LOGGER = logging.getLogger('slice_lid.train')

def return_training_stats(train_log, savedir):
    """Return a dict with a summary of training results.

    Parameters
    ----------
    train_log : keras.History
        Training history returned by `keras.model.fit`
    savedir : str
        Directory where trained model is saved.

    Return
    ------
    dict
        Dictionary with training summary.
    """

    best_idx = np.argmin(train_log.history['val_categorical_accuracy_1'])

    result = {
        'val_loss'  : train_log.history['val_loss'][best_idx],
        'val_acc '  : train_log.history['val_categorical_accuracy'][best_idx],
        'val_wacc ' :
            train_log.history['val_categorical_accuracy_1'][best_idx],
        'savedir'   : savedir,
    }

    return result

def create_and_train_model(args = None, extra_kwargs = None, **kwargs):
    """Creates and trains `keras` model specified by arguments.

    Parameters
    ----------
    args : Args or None, optional
        Specification of the model and training setup
        If None, then the model and training specification will be first
        constructed from `kwargs` and `extra_kwargs`
    extra_kwargs : dict or None, optional
        Extra kwargs that will be passed to the `Args` constructor.
    kwargs : dict
        Parameters that will be passed to the `Args` constructor if `args` is
        None.

    Return
    ------
    dict
        Dictionary with training summary returned by `return_training_stats`.

    See Also
    --------
    slice_lid.args.Args
    return_training_stats
    """

    if args is None:
        args = Args(extra_kwargs = extra_kwargs, **kwargs)

    LOGGER.info(
        "Starting training with parameters:\n%s", args.config.pprint()
    )

    LOGGER.info("Loading data...")
    dgen_train, dgen_test = load_data(args)

    LOGGER.info("Creating model...")
    np.random.seed(args.seed)

    model     = select_model(args)
    callbacks = get_default_callbacks(args)
    optimizer = get_optimizer(args.optimizer)

    LOGGER.info("Compiling model...")
    model.compile(
        loss             = 'categorical_crossentropy',
        metrics          = [ 'categorical_accuracy' ],
        weighted_metrics = [
            'categorical_accuracy', 'categorical_crossentropy'
        ],
        optimizer        = optimizer,
    )

    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, len(dgen_train))

    LOGGER.info("Training model...")
    train_log = model.fit_generator(
        dgen_train,
        epochs          = args.epochs,
        validation_data = dgen_test,
        callbacks       = callbacks,
        steps_per_epoch = steps_per_epoch,
        shuffle         = True,
        **get_keras_concurrency_kwargs(args)
    )

    LOGGER.info("Training Complete")

    return return_training_stats(train_log, args.savedir)

