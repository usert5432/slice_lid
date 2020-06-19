"""Functions to calculate class weights"""

import numpy as np

def equal_class_weights(targets):
    """Calculate weights that will make targets equally represented"""
    n_classes = targets.shape[1]
    counts    = np.zeros(n_classes)

    class_idx = targets.argmax(axis = 1)
    np.add.at(counts, class_idx, 1)

    # NOTE: in order to preserve total normalization
    #           sum_over_classes[c] of (N[c] * W[c]) == N
    #       class weights are defined by
    #           W[c] = (sum[x] of N[x]) / (n_classes * N[c])
    weights = np.sum(counts) / (n_classes * counts)

    return weights

def calc_class_weights(weights, targets):
    """Calculate class weights.

    Parameters
    ----------
    weights : { 'equal', None }
        Name of the class weights. If None, then no class weights will be
        calculated.
        If 'equal' then the weights that make targets equally represented
        will be returned.
    targets : ndarray, shape (N, N_TARGET)
        Array of one-hot target encodings.

    Returns
    -------
    ndarray, shape (N,) or None
        Array of weights, or None if `weights` is None.
    """

    if weights is None:
        return None

    if weights == 'equal':
        return equal_class_weights(targets)

    raise ValueError("Unknown class_weights: %s" % (weights))

