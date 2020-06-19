"""Functions to make plots of PID histograms"""

import os
import matplotlib.pyplot as plt

from cafplot.rhist.rhist1d import RHist1D
from cafplot.plot          import plot_rhist1d, plot_rhist1d_error, save_fig

def plot_single_distribution(
    ax, truth_idx, truth, preds, weights, bins, labels, **kwargs
):
    """Plot distribution of predicted PID values for a single true component.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes where histogram will be plotted.
    truth_idx : int
        Value of the true component to be plotted.
    truth : ndarray, shape (N_SAMPLES,)
        Array of true targets.
    preds : ndarray, shape (N_SAMPLES,)
        Array of predicted target scores.
    weights : ndarray, shape (N_SAMPLES,)
        Sample weights.
    bins : int
        Number of bins of the hist plots.
    labels : list of str, len(N_TARGETS)
        List of labels for each target.
    kwargs : dict
        Parameters that will be passed to the histogram plot functions.

    See Also
    --------
    plot_rhist1d
    plot_rhist1d_error
    """

    truth_mask = (truth == truth_idx)

    rhist = RHist1D.from_data(
        preds[truth_mask], weights = weights[truth_mask],
        bins = bins, range = (0, 1)
    )

    plot_rhist1d(ax, rhist, labels[truth_idx], histtype = 'step', **kwargs)
    plot_rhist1d_error(
        ax, rhist, err_type = 'bar', err = 'normal', sigma = 1, **kwargs
    )

def plot_detailed_distributions(
    truth, preds, weights, bins, labels, log_scale, x_label
):
    """Plot distribution of predicted PID values for each true component.

    Parameters
    ----------
    truth : ndarray, shape (N_SAMPLES,)
        Array of true targets.
    preds : ndarray, shape (N_SAMPLES,)
        Array of predicted target scores.
    weights : ndarray, shape (N_SAMPLES,)
        Sample weights.
    bins : int
        Number of bins of the hist plots.
    labels : list of str, len(N_TARGETS)
        List of labels for each target.
    log_scale : bool
        If True then the y-axis will have log-scale.
    x_label : str
        Label of the x axis.

    Returns
    -------
    f : matplotlib.Figure
        Matplotlib figure.
    ax : matplotlib.Axes
        Matplotlib Axes.
    """

    f, ax = plt.subplots()

    if log_scale:
        ax.set_yscale('log')

    for truth_idx in range(len(labels)):
        plot_single_distribution(
            ax, truth_idx, truth, preds, weights, bins, labels,
            color = 'C%d' % (truth_idx, )
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Events")
    ax.set_xlim((0, 1))

    ax.legend()

    return f, ax

def plot_distributions(truth, preds, weights, bins, labels, plotdir, ext):
    """Make and save plots of hists of pred PID values for each true component.

    Parameters
    ----------
    truth : ndarray, shape (N_SAMPLES,)
        Array of true targets.
    preds : ndarray, shape (N_SAMPLES, N_TARGETS)
        Array of predicted target scores.
    weights : ndarray, shape (N_SAMPLES,)
        Sample weights.
    bins : int
        Number of bins of the hist plots.
    labels : list of str, len(N_TARGETS)
        List of labels for each target.
    plotdir : str
        Directory where plots will be saved.
    ext : str or list of str
        Extension of the plots. If list then the plots will be saved in
        multiple formats.
    """

    for pred_idx,x_label in enumerate(labels):
        for log_scale in [ True, False ]:
            label = labels[pred_idx]

            f, _ax = plot_detailed_distributions(
                truth, preds[:, pred_idx], weights, bins, labels, log_scale,
                x_label
            )

            fname = 'distrib_%s_log(%s)' % (label, log_scale)
            save_fig(f, os.path.join(plotdir, fname), ext)

