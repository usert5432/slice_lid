"""Calculate and plot target frequencies in a sample."""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from cafplot.rhist      import RHist1D
from cafplot.plot       import plot_rhist1d, make_plotdir, save_fig
from lstm_ee.utils      import setup_logging
from lstm_ee.utils.eval import make_eval_outdir

from slice_lid.args              import Args
from slice_lid.data              import load_data
from slice_lid.plot.labels       import convert_targets_to_labels
from slice_lid.utils.parsers     import add_basic_eval_args
from slice_lid.utils.eval_config import EvalConfig

def parse_cmdargs():
    # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser("Plot sample targets distribution")
    add_basic_eval_args(parser)
    return parser.parse_args()

def count_events(dgen):
    # pylint: disable=missing-function-docstring
    targets = dgen.get_target_data(None)
    weights = dgen.weights
    counts  = np.dot(weights, targets)

    return counts

def plot_counts(counts, labels, plotdir, ext):
    # pylint: disable=missing-function-docstring
    norm = np.sum(counts)

    f, ax = plt.subplots()

    rhist = RHist1D(np.arange(len(labels) + 1), counts)
    rhist.scale(100 / norm)

    plot_rhist1d(ax, rhist, label = "", histtype = 'bar', color = 'tab:blue')

    ax.set_xticks([ i + 0.5 for i in range(len(labels))])
    ax.set_xticklabels(labels)

    ax.set_ylabel("Fraction [%]")
    ax.set_title("Distribution of events per category")

    save_fig(f, os.path.join(plotdir, "sample_distribution_test"), ext)

def main():
    # pylint: disable=missing-function-docstring
    setup_logging()
    cmdargs = parse_cmdargs()

    args = Args.load(savedir = cmdargs.outdir)
    eval_config = EvalConfig.from_cmdargs(cmdargs)

    eval_config.modify_eval_args(args)

    _, dgen = load_data(args)
    outdir  = make_eval_outdir(cmdargs.outdir, eval_config)
    plotdir = make_plotdir(outdir)

    counts = count_events(dgen)
    labels = convert_targets_to_labels(args.target_pdg_iscc_list)

    plot_counts(counts, labels, plotdir, cmdargs.ext)

if __name__ == '__main__':
    main()

