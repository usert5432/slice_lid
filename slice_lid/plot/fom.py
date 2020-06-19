"""Functions to make plots of various Figures Of Merit."""

import os
import matplotlib.pyplot as plt
import numpy as np

from cafplot.plot import plot_rhist1d, save_fig

def decorate_fom_axes(ax, x_label):
    """Adjust style of FOM plot axes"""
    ax.set_xlim((0, 1))

    ax.set_xlabel(x_label)
    ax.minorticks_on()

    ax.grid(True, which = 'major', linestyle = 'dashed', linewidth = 1.0)
    ax.grid(True, which = 'minor', linestyle = 'dashed', linewidth = 0.5)

def plot_single_fom(rhist_fom, label, fom_label, **kwargs):
    """Make a plot of a single FOM distribution

    Parameters
    ----------
    rhist_fom : cafplot.RHist1D
        Histogram containing FOM.
    label : str
        Plot label.
    fom_label : str
        Name of the Figure of Merit.
    kwargs : dict
        Parameters that will be passed to `plot_rhist1d` function.

    Returns
    -------
    f : matplotlib.Figure
        Matplotlib figure.
    ax : matplotlib.Axes
        Matplotlib Axes.
    """
    f, ax = plt.subplots()

    plot_rhist1d(ax, rhist_fom, None, histtype = 'step', **kwargs)
    decorate_fom_axes(ax, label)

    ax.set_title('%s for %s' % (fom_label.capitalize(), label.capitalize()))
    ax.set_ylabel(fom_label.capitalize())

    return f, ax

def plot_maxval_line(ax, max_pos, max_val, **kwargs):
    """Plot vertical line indicating position of the maximum FOM value"""
    label = 'Max Value: %.3e, Position: %.3e' % (max_val, max_pos)
    ax.axvline(max_pos, 0, 1, label = label, **kwargs)

def find_fom_maxval(rhist_fom):
    """Find maximum FOM value and its position."""
    max_idx = np.argmax(rhist_fom.hist)
    max_val = rhist_fom.hist[max_idx]
    max_pos = (rhist_fom.bins[max_idx] + rhist_fom.bins[max_idx+1]) / 2

    return (max_pos, max_val)

def plot_separate_foms(rhist_fom_list, labels, fom_label, plotdir, ext):
    """Make and save separate plots of FOMs.

    Parameters
    ----------
    rhist_fom_list : list of cafplot.RHist1D
        List of figures of merit to be plotted.
    labels : list of str
        List of x axis labels. One for each item in `rhist_fom_list`.
    fom_label : str
        Name of the FOM.
    plotdir : str
        Directory where plots will be saved.
    ext : str or list of str
        Extension of the plots. If list then the plots will be saved in
        multiple formats.
    """

    for pred_idx,x_label in enumerate(labels):
        rhist_fom = rhist_fom_list[pred_idx]

        f, ax = plot_single_fom(rhist_fom, x_label, fom_label, color = 'C0')

        max_pos, max_val = find_fom_maxval(rhist_fom)
        plot_maxval_line(ax, max_pos, max_val, color = 'C1')

        ax.legend()

        fname = 'fom_%s_%s' % (fom_label, x_label)
        save_fig(f, os.path.join(plotdir, fname), ext)
        plt.close(f)

def plot_overlayed_foms(fom_dict, foms_to_overlay, labels, plotdir, ext):
    """Make and save separate plots of FOMs.

    Multiple plots will be made -- one for each target.
    Multiple FOMs will be overlayed on a single plot -- one for each element
    in `foms_to_overlay`.

    Parameters
    ----------
    fom_dict : dict
        Dictionary of FOMs where keys are names of the FOM and values are
        lists of cafplot.RHist1D containing FOM for different targets.
    foms_to_overlay : list of str
        List of FOM names that will be overlayed on a single plot.
    labels : list of str
        List of target names. Each list in the `fom_dict` should have the
        same length as `labels`.
    plotdir : str
        Directory where plots will be saved.
    ext : str or list of str
        Extension of the plots. If list then the plots will be saved in
        multiple formats.
    """

    for pred_idx,x_label in enumerate(labels):
        f, ax = plt.subplots()

        for fom_idx, fom_label in enumerate(foms_to_overlay):
            rhist_fom = fom_dict[fom_label][pred_idx]

            rhist_fom.scale(1 / np.max(rhist_fom.hist))
            plot_rhist1d(
                ax, rhist_fom, fom_label.capitalize(), histtype = 'step',
                color = 'C%d' % (fom_idx,)
            )

        decorate_fom_axes(ax, x_label)

        ax.set_ylabel('FOMs')
        ax.set_title('Normalized FOMs')
        ax.legend()

        fname = 'overlayed_foms_%s' % (x_label,)
        save_fig(f, os.path.join(plotdir, fname), ext)
        plt.close(f)

