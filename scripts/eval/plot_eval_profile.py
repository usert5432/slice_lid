"""Plot error matrix components vs model configuration"""

import argparse
import os

from slice_lid.args import Args
from slice_lid.eval.error_matrix import (
    load_error_matrix, normalize_error_matrix
)
from slice_lid.plot.profile import plot_profile


def parse_cmdargs():
    # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser(
        "Plot model performance vs hyperparameters"
    )

    parser.add_argument(
        '-v', '--var', nargs = '+', required = True, type = str,
        help = 'Variable that is changed between models'
    )

    parser.add_argument(
        '--eval_dir', dest = 'eval_dir', required = True, type = str,
        help = 'Directory with error matrices'
    )

    parser.add_argument(
        'models', metavar = 'OUTDIR', nargs = '+', type = str,
        help = 'Directory with saved models'
    )

    parser.add_argument(
        '-e', '--ext', default = [ 'png' ],  dest = 'ext', nargs = '+',
        type = str, help='Plot file extension'
    )

    parser.add_argument(
        '-c', '--categorical', action = 'store_true', dest = 'categorical',
        help = 'Directory with saved models'
    )

    parser.add_argument(
        '-s', '--sort', choices = [ 'x', 'y', None ], default = None,
        dest = 'sort', type = str, help = 'Sort values'
    )

    parser.add_argument(
        '-o', '--outdir', dest = 'outdir', required = True, type = str,
        help = 'Directory to save plots'
    )

    return parser.parse_args()

def load_model_err_matrices(var, models, eval_dir):
    # pylint: disable=missing-function-docstring
    var_list             = []
    err_mat_list         = []
    target_pdg_iscc_list = None

    for savedir in models:
        try:
            args    = Args.load(savedir = savedir)
            err_mat = load_error_matrix(
                os.path.join(savedir, eval_dir)
            )

            if target_pdg_iscc_list is None:
                target_pdg_iscc_list = args.target_pdg_iscc_list
            else:
                assert(target_pdg_iscc_list == args.target_pdg_iscc_list)

        except IOError:
            print("Failed to load model: %s" % savedir)
            continue

        if var == 'label':
            with open(os.path.join(savedir, "label"), "rt") as f:
                var_list.append(f.readline())
        else:
            var_list.append(args[var])

        err_mat_list.append(err_mat)

    return var_list, err_mat_list, target_pdg_iscc_list

def main():
    # pylint: disable=missing-function-docstring
    cmdargs = parse_cmdargs()

    if len(cmdargs.var) == 1:
        cmdargs.var = cmdargs.var[0]

    var_list, err_mat_list, target_pdg_iscc_list = load_model_err_matrices(
        cmdargs.var, cmdargs.models, cmdargs.eval_dir
    )

    assert(len(var_list) > 0)

    os.makedirs(cmdargs.outdir, exist_ok = True)

    for by_truth,label_y in [
        (True,  'Recall'),
        (False, 'Precision'),
    ]:

        err_matrices = [
            normalize_error_matrix(x, by_truth) for x in err_mat_list
        ]

        fname = os.path.join(
            cmdargs.outdir,
            "%s_%s_sort(%s)_cat(%s)" % (
                label_y.lower(), cmdargs.var, cmdargs.sort, cmdargs.categorical
            )
        )

        plot_profile(
            var_list, err_matrices, target_pdg_iscc_list,
            label_x     = cmdargs.var,
            label_y     = label_y,
            sort_type   = cmdargs.sort,
            categorical = cmdargs.categorical,
            fname       = fname,
            ext         = cmdargs.ext
        )

if __name__ == '__main__':
    main()

