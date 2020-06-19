"""
Functions to make plots of error (confusion) matrices.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from cafplot.plot import save_fig
from slice_lid.eval.error_matrix import normalize_error_matrix

def pplot_matrix_values(ax, mat):
    """Print error matrix values in readable color"""
    color_threshold = (np.nanmax(mat) + np.nanmin(mat)) / 2

    for i in range(len(mat)):
        for j in range(len(mat[0])):
            color = "white" if mat[i, j] < color_threshold else "black"
            ax.text(
                j, i,
                "%.1f" % (100 * mat[i, j]),
                horizontalalignment = "center",
                verticalalignment   = "center",
                color               = color
            )

def plot_error_matrix_base(err_mat, labels, by_truth, fname, ext):
    """Make and save plot of a single error matrix

    Parameters
    ----------
    err_mat : ndarray, shape (N, N)
        Error matrix. First axis is true dimension, second is predicted.
    labels : list of str, len(N)
        List of labels for each target of the error matrix `err_mat`.
    by_truth : bool
        Indicates whether error matrix `err_mat` is normalized by true values.
        If False, then it is assumed that the error matrix is normalized by
        predicted values.
    fname : str
        File path without extension where the error matrix will be saved.
    ext : str or list of str
        Extension of the plot. If list then the plot will be saved in multiple
        formats.
    """
    f, ax = plt.subplots()

    im = ax.imshow(err_mat)

    pplot_matrix_values(ax, err_mat)

    ax.set_xlabel("Prediction")
    ax.set_ylabel("Truth")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, rotation='vertical', va='center')

    if by_truth is not None:
        if by_truth:
            ax.set_title('Normalized by Truth')
        else:
            ax.set_title('Normalized by Preds')

    f.colorbar(im)
    save_fig(f, fname, ext)

def plot_error_matrix(err_mat, labels, plotdir, ext):
    """Make and save plots of an error matrix with different normalizations.

    Parameters
    ----------
    err_mat : ndarray, shape (N, N)
        Error matrix. First axis is true dimension, second is predicted.
    labels : list of str, len(N)
        List of labels for each target of the error matrix `err_mat`.
    plotdir : str
        Directory where plots will be saved.
    ext : str or list of str
        Extension of the plots. If list then the plots will be saved in
        multiple formats.
    """

    plot_error_matrix_base(
        err_mat, labels, None, os.path.join(plotdir, "err_mat"), ext
    )

    for (by_truth,label) in [ (True, 'truth'), (False, 'preds') ]:
        norm_err_mat = normalize_error_matrix(err_mat, by_truth)

        plot_error_matrix_base(
            norm_err_mat, labels, by_truth,
            os.path.join(plotdir, "err_mat_normed_%s" % (label)),
            ext
        )

