"""
A collection of evaluation routines.
"""

import os
import numpy as np

from cafplot.plot.funcs import make_plotdir
from lstm_ee.utils.eval import make_eval_outdir, modify_concurrency_args

from slice_lid.args import Args
from slice_lid.data import load_data
from .eval_config   import EvalConfig
from .io            import load_model

DEFAULT_RECO_MAP = {
    None    : 'cvn.ncid',
    (12, 1) : 'cvn.nueid',
    (14, 1) : 'cvn.numuid',
    (16, 1) : 'cvn.nutauid',
    (0,  0) : 'cvn.cosmicid',
}

def standard_eval_prologue(cmdargs):
    """Standard evaluation prologue"""
    args, model = load_model(cmdargs.outdir, compile = False)
    eval_config = EvalConfig.from_cmdargs(cmdargs)

    eval_config.modify_eval_args(args)
    modify_concurrency_args(args, cmdargs)

    _, dgen    = load_data(args)
    outdir     = make_eval_outdir(cmdargs.outdir, eval_config)
    plotdir    = make_plotdir(outdir)

    return (dgen, args, model, outdir, plotdir)

def get_reco_preds(args, dgen, reco_map = None):
    """Get reco values from the dataset

    Parameters
    ----------
    args : Args
        Parameters of the network.
    dgen : IDataGenerator
        Dataset.
    reco_map : dict
        Dictionary where keys are (int, bool) pairs specifying (pdg, iscc)
        values and targets are the variable names in `dgen` corresponding
        to the reconstructed values for these (pdg, iscc) pairs.

    Returns
    -------
    ndarray, shape (len(dgen.data_loader), len(reco_map))
        Reconstructed values loaded from the dataset.
    """

    if reco_map is None:
        reco_map = DEFAULT_RECO_MAP

    reco_pred_map = { k : dgen.data_loader.get(v) for k,v in reco_map.items() }

    result = {}
    result[None] = reco_pred_map[None]

    targets = [ tuple(x) for x in args.target_pdg_iscc_list ]

    for k,v in reco_pred_map.items():
        if k is None:
            continue
        if k in targets:
            result[k] = v
        else:
            result[None] += v

    return np.vstack([ result[None], ] + [ result[k] for k in targets ]).T

def reco_eval_prologue(cmdargs, reco_map = None):
    """Evaluation prologue that loads reco values from the dataset"""
    args = Args.load(cmdargs.outdir)

    eval_config = EvalConfig.from_cmdargs(cmdargs)
    eval_config.modify_eval_args(args)

    _, dgen    = load_data(args)
    outdir     = make_eval_outdir(cmdargs.outdir, eval_config)
    outdir     = os.path.join(outdir, 'reco(%s)' % (reco_map))
    plotdir    = make_plotdir(outdir)

    reco_preds = get_reco_preds(args, dgen, reco_map)

    return (dgen, args, reco_preds, outdir, plotdir)

