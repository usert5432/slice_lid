"""Make plot of the t-SNE embedding of the predicted scores."""

import argparse
import os
import sklearn.manifold

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from cafplot.plot  import save_fig
from lstm_ee.utils import setup_logging
from lstm_ee.utils.parsers import add_concurrency_parser

from slice_lid.eval.distribution import get_truth_preds_arrays
from slice_lid.plot.labels       import convert_targets_to_labels
from slice_lid.utils.parsers     import add_basic_eval_args
from slice_lid.utils.eval        import standard_eval_prologue

def parse_cmdargs():
    # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser("Make t-SNE embedding plots")

    parser.add_argument(
        '-b', '--bins',
        dest    = 'bins',
        default = 100,
        help    = 'Number of bins for the t-SNE density plot',
        type    = int,
    )

    parser.add_argument(
        '-n', '--number',
        dest    = 'limit',
        default = 2000,
        help    = 'Number of points for the t-SNE embedding',
        type    = int,
    )

    parser.add_argument(
        '--dimensions',
        dest    = 'dims',
        default = 2,
        choices = [ 2, 3 ],
        help    = 'Number of dimensions of embedding manifold',
        type    = int,
    )

    parser.add_argument(
        '--perplexity',
        dest    = 'perplexity',
        default = 30,
        help    = 'Perplexity value for the t-SNE embedding',
        type    = float,
    )

    add_basic_eval_args   (parser)
    add_concurrency_parser(parser)

    return parser.parse_args()

def calc_embedding(data, perplexity, dims):
    """Calculate t-SNE embedding coordinates"""
    tsne = sklearn.manifold.TSNE(
        perplexity = perplexity, n_components = dims, verbose = 10
    )

    return tsne.fit_transform(data)

def plot_2d_embedding_scatter(truth, preds, labels, fname_base, ext):
    """Make a scatterplot of embedded data"""
    classes = np.arange(len(labels))

    f, ax = plt.subplots()

    # To make points visible and avoid clutter
    if len(truth) >= 20000:
        alpha = 0.25
    elif len(truth) >= 10000:
        alpha = 0.50
    else:
        alpha = 0.75

    for class_idx in classes:
        truth_mask = (truth == class_idx)
        preds_idx  = preds[truth_mask, :]
        color      = 'C%d' % (class_idx,)

        ax.scatter(
            preds_idx[:, 0], preds_idx[:, 1], label = labels[class_idx],
            marker = ',', color = color, alpha = alpha
        )

    add_nice_legend(ax)
    save_fig(f, fname_base + "_scatter", ext)

def normalize_alpha_channel(hist2d_rgba, cmin, cmax):
    """Normalize alpha channel of 2D array of RGBA colors."""
    null_mask = (hist2d_rgba[...,3] == 0)

    hist2d_rgba[...,3] = hist2d_rgba[...,3] / np.max(hist2d_rgba[...,3])
    hist2d_rgba[...,3] = cmin + (cmax - cmin) * hist2d_rgba[...,3]
    hist2d_rgba[...,3][null_mask] = 0

def mix_density_hists(hists):
    """Mix 2D arrays of RGBA colors.

    This function receives a list of 2D arrays of RGBA colors as input.
    It mixes these arrays together to produce a single 2D array of mixed RGBA
    colors.
    """
    result    = np.zeros(hists[0].shape + (4,))
    alpha_sum = np.zeros(hists[0].shape)

    result[...,3] = 1

    for idx,hist in enumerate(hists):
        alpha = mpl.colors.Normalize(vmin = 0, vmax = 1)(hist)
        color = mpl.colors.to_rgb('C%d' % (idx,))

        result[...,0] += alpha * color[0]
        result[...,1] += alpha * color[1]
        result[...,2] += alpha * color[2]
        result[...,3] *= (1 - alpha)

        alpha_sum     += alpha

    result[...,0] /= alpha_sum
    result[...,1] /= alpha_sum
    result[...,2] /= alpha_sum
    result[...,3] = 1 - result[...,3]

    # make colors more bright
    normalize_alpha_channel(result, 0.5, 1)

    return result

def find_values_range(values, margin = 0.05):
    # pylint: disable=missing-function-docstring
    a = np.min(values)
    b = np.max(values)

    values_range = (b - a)

    return [a - margin * values_range, b + margin * values_range]

def get_color_hist(truth, preds, labels, bins):
    """Get a 2D array of RGBA colors representing density of true classes.

    Parameters
    ----------
    truth : ndarray, shape (N_SAMPLES,)
        True labels (classes) of samples.
    preds : ndarray, shape (N_SAMPLES, 2)
        Embedded coordinates of predicted scores of targets (classes).
    labels : list of str
        Names of the classes.
    bins : int
        Number of bins in each histogram dimension.

    Returns
    -------
    mixed_hist : ndarray, shape (`bins`, `bins`, 4)
        2D array of RGBA colors representing density of classes.
    xbins : ndarray, shape (`bins+1, )
        Bin edges in the x coordinate.
    ybins : ndarray, shape (`bins+1, )
        Bin edges in the y coordinate.
    """
    classes = np.arange(len(labels))

    xlim  = find_values_range(preds[:,0])
    ylim  = find_values_range(preds[:,1])
    hists = []

    for class_idx in classes:
        mask = (truth == class_idx)

        class_hist, xbins, ybins = np.histogram2d(
            preds[mask, 0], preds[mask, 1],
            bins    = bins,
            range   = [xlim, ylim],
            density = True
        )
        class_hist[np.isnan(class_hist)] = 0
        hists.append(class_hist)

    mixed_hist = mix_density_hists(hists)

    return (mixed_hist, xbins, ybins)

def add_nice_legend(ax, **kwargs):
    """Add legend on top of plot"""
    ax.legend(
        bbox_to_anchor = (0.0, 1.01, 1.0, 1.11), fancybox = True,
        loc = 'lower left', mode = 'expand', ncol = 5,
        **kwargs
    )

def plot_2d_embedding_density(truth, preds, labels, bins, fname_base, ext):
    """Make a density plot of embedded coordinates."""
    color_hist, xbins, ybins = get_color_hist(truth, preds, labels, bins)
    f, ax = plt.subplots()

    ax.imshow(
        np.transpose(color_hist, (1, 0, 2)),
        interpolation = 'nearest',
        origin        = 'low',
        extent        = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
    )

    ax.set_aspect('auto')

    handles = [
        mpatches.Patch(color = 'C%d' % (idx), label = label)
            for idx,label in enumerate(labels)
    ]

    add_nice_legend(ax, handles = handles)
    save_fig(f, fname_base + "_density", ext)

def main():
    # pylint: disable=missing-function-docstring
    setup_logging()
    cmdargs = parse_cmdargs()

    dgen, args, model, _outdir, plotdir = standard_eval_prologue(cmdargs)

    truth_array, preds_array = get_truth_preds_arrays(dgen, model)
    labels = convert_targets_to_labels(args.target_pdg_iscc_list)

    truth = truth_array[:cmdargs.limit]
    preds = preds_array[:cmdargs.limit, :]

    embedded_preds = calc_embedding(preds, cmdargs.perplexity, cmdargs.dims)

    fname_base = 'tsne_%d_perp(%e)_lim(%d)' % (
        cmdargs.dims, cmdargs.perplexity, cmdargs.limit
    )

    fname_base = os.path.join(plotdir, fname_base)

    if cmdargs.dims == 2:
        plot_2d_embedding_scatter(
            truth, embedded_preds, labels, fname_base, cmdargs.ext
        )
        plot_2d_embedding_density(
            truth, embedded_preds, labels, cmdargs.bins, fname_base,
            cmdargs.ext
        )

if __name__ == '__main__':
    main()

