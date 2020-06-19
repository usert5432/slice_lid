"""Functions to calculate PID distributions"""

import numpy as np
from cafplot.rhist import RHist1D

def get_truth_preds_arrays(dgen, model):
    """Get true and predicted targets.

    Parameters
    ----------
    dgen : IDataGenerator
        Dataset.
    Model : keras.Model
        Network

    Returns
    -------
    truth : ndarray, shape (len(dgen.data_loader),)
        Array of true targets.
    preds : ndarray, shape (len(dgen.data_loader), N_TARGETS)
        Array of predicted target scores.
    """

    accum_truth = []
    accum_preds = []

    for batch in dgen:
        inputs  = batch[0]
        targets = batch[1]

        preds = model.predict(inputs)
        truth = targets['target'].argmax(axis = 1).ravel()

        accum_truth.append(truth)
        accum_preds.append(preds)

    return (np.hstack(accum_truth), np.vstack(accum_preds))

def get_sgn_bkg_preds(truth, preds, weights, truth_idx, pred_idx, **kwargs):
    """Calculate Signal and Background histograms.

    Parameters
    ----------
    truth : ndarray, shape (N_SAMPLES,)
        Array of true targets.
    preds : ndarray, shape (N_SAMPLES, N_TARGETS)
        Array of predicted target scores.
    weights : ndarray, shape (N_SAMPLES,)
        Sample weights.
    truth_idx : int
        Value of `truth` target that indicates signal sample.
    pred_idx : int
        Index of the target to make histogram of.
    kwargs : dict
        Dictionary of values that will be passed to the `RHist1D` constructor.

    Returns
    -------
    h_sgn : RHist1D
        Histogram of signal values.
    h_bkg : RHist1D
        Histogram of background values.
    """

    sgn_mask = (truth == truth_idx)

    preds_sgn = preds[sgn_mask,  pred_idx]
    preds_bkg = preds[~sgn_mask, pred_idx]

    w_sgn = weights[sgn_mask]
    w_bkg = weights[~sgn_mask]

    return (
        RHist1D.from_data(preds_sgn, weights = w_sgn, **kwargs),
        RHist1D.from_data(preds_bkg, weights = w_bkg, **kwargs),
    )

