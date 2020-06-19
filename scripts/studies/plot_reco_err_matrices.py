"""Make plots of baseline error matrices"""
import argparse
import numpy as np

from lstm_ee.utils               import setup_logging
from slice_lid.eval.error_matrix import save_error_matrix
from slice_lid.plot.error_matrix import plot_error_matrix
from slice_lid.plot.labels       import convert_targets_to_labels
from slice_lid.utils.parsers     import add_basic_eval_args
from slice_lid.utils.eval        import reco_eval_prologue

def parse_cmdargs():
    # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser("Make error matrix plots for reco")
    add_basic_eval_args(parser)
    return parser.parse_args()

def calc_reco_err_matrix(preds, dgen):
    # pylint: disable=missing-function-docstring
    preds_labels = preds.argmax(axis = 1)
    truth_labels = dgen.get_target_data(None).argmax(axis = 1)

    err_mat = np.zeros((preds.shape[1], preds.shape[1]))
    np.add.at(err_mat, (truth_labels, preds_labels), 1)

    return err_mat

def main():
    # pylint: disable=missing-function-docstring
    setup_logging()
    cmdargs = parse_cmdargs()

    dgen, args, reco_preds, outdir, plotdir = reco_eval_prologue(cmdargs)

    err_mat = calc_reco_err_matrix(reco_preds, dgen)
    save_error_matrix(outdir, err_mat)

    labels = convert_targets_to_labels(args.target_pdg_iscc_list)
    plot_error_matrix(err_mat, labels, plotdir, cmdargs.ext)

if __name__ == '__main__':
    main()

