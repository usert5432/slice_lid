"""Functions to calculate error (confusion) matrices"""

import os
import numpy as np

def make_error_matrix(dgen, model, targets_pdg_iscc_list):
    """Calculate error matrix.

    Parameters
    ----------
    dgen : IDataGenerator
        Dataset.
    model : keras.Model
        Network.
    targets_pdg_iscc_list : list of (int, bool)
        List of targets. C.f. `Config.targets_pdg_iscc_list`.

    Returns
    -------
    ndarray, shape (N_TARGET, N_TARGET)
        Error matrix where first dimension corresponds to the true values and
        the second dimension correspond to the predicted values.
        N_TARGET = len(`targets_pdg_iscc_list`) + 1
    """

    n_targets = len(targets_pdg_iscc_list) + 1
    err_mat   = np.zeros((n_targets, n_targets))

    for batch in dgen:
        inputs  = batch[0]
        targets = batch[1]

        preds = model.predict(inputs).argmax(axis = 1)
        truth = targets['target'].argmax(axis = 1)

        np.add.at(err_mat, (truth, preds), 1)

    return err_mat

def normalize_error_matrix(err_mat, by_truth = True):
    """Normalize error matrix either by truth or by preds axes"""

    if by_truth:
        norm = err_mat.sum(axis = 1, keepdims = 1)
    else:
        norm = err_mat.sum(axis = 0, keepdims = 1)

    return err_mat / norm

def save_error_matrix(outdir, error_matrix):
    """Save error matrix to a file"""
    fname = os.path.join(outdir, "err_mat.txt")
    np.savetxt(fname, error_matrix)

def load_error_matrix(outdir):
    """Load error matrix from a file"""
    fname = os.path.join(outdir, "err_mat.txt")
    return np.loadtxt(fname)

