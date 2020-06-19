"""Calculate error matrices for a trained model."""

import argparse

from lstm_ee.utils         import setup_logging
from lstm_ee.utils.parsers import add_concurrency_parser

from slice_lid.eval.error_matrix import make_error_matrix, save_error_matrix
from slice_lid.plot.error_matrix import plot_error_matrix
from slice_lid.plot.labels       import convert_targets_to_labels
from slice_lid.utils.parsers     import add_basic_eval_args
from slice_lid.utils.eval        import standard_eval_prologue

def parse_cmdargs():
    # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser("Make error matrix plots")
    add_basic_eval_args(parser)
    add_concurrency_parser(parser)

    return parser.parse_args()

def main():
    # pylint: disable=missing-function-docstring
    setup_logging()
    cmdargs = parse_cmdargs()

    dgen, args, model, outdir, plotdir = standard_eval_prologue(cmdargs)

    err_mat = make_error_matrix(dgen, model, args.target_pdg_iscc_list)
    save_error_matrix(outdir, err_mat)

    labels = convert_targets_to_labels(args.target_pdg_iscc_list)
    plot_error_matrix(err_mat, labels, plotdir, cmdargs.ext)

if __name__ == '__main__':
    main()

