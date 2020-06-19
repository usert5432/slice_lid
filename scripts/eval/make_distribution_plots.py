"""Make plots of predicted target score distributions"""

import argparse

from lstm_ee.utils         import setup_logging
from lstm_ee.utils.parsers import add_concurrency_parser

from slice_lid.eval.distribution import get_truth_preds_arrays
from slice_lid.plot.labels       import convert_targets_to_labels
from slice_lid.plot.distribution import plot_distributions
from slice_lid.utils.parsers     import add_basic_eval_args
from slice_lid.utils.eval        import standard_eval_prologue

def parse_cmdargs():
    # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser(
        "Make plots of target score distributions"
    )
    parser.add_argument(
        '-b', '--bins',
        dest    = 'bins',
        default = 50,
        help    = 'Number of bins in distribution plots',
        type    = int,
    )
    add_basic_eval_args(parser)
    add_concurrency_parser(parser)

    return parser.parse_args()

def main():
    # pylint: disable=missing-function-docstring
    setup_logging()
    cmdargs = parse_cmdargs()

    dgen, args, model, _outdir, plotdir = standard_eval_prologue(cmdargs)

    truth_array, preds_array = get_truth_preds_arrays(dgen, model)
    labels = convert_targets_to_labels(args.target_pdg_iscc_list)

    plot_distributions(
        truth_array, preds_array, dgen.weights, cmdargs.bins, labels,
        plotdir, cmdargs.ext
    )

if __name__ == '__main__':
    main()

