"""Functions to evaluate figures of merit"""

import numpy as np
from cafplot.rhist import RHist1D
from .distribution import get_sgn_bkg_preds

def fom_efficiency(sgn_cumsum, _bkg_cumsum):
    """Efficiency"""
    return sgn_cumsum.hist / sgn_cumsum.hist[0]

def fom_purity(sgn_cumsum, bkg_cumsum):
    """Purity"""
    return sgn_cumsum.hist / (sgn_cumsum.hist + bkg_cumsum.hist)

def fom_selection(sgn_cumsum, bkg_cumsum):
    """FOM = S / sqrt(S + B)"""
    return sgn_cumsum.hist / np.sqrt((sgn_cumsum.hist + bkg_cumsum.hist))

FOM_SPEC_DICT = {
    'efficiency' : fom_efficiency,
    'purity'     : fom_purity,
    'selection'  : fom_selection,
}

def rhist1d_cumsum(rhist, reverse = True):
    """Calculate cumulative sum of a histogram"""

    if not reverse:
        cum_hist   = np.cumsum(rhist.hist)
        cum_err_sq = np.cumsum(rhist.err_sq)
    else:
        cum_hist   = np.cumsum(rhist.hist[::-1]  )[::-1]
        cum_err_sq = np.cumsum(rhist.err_sq[::-1])[::-1]

    return RHist1D(rhist.bins, cum_hist, cum_err_sq)

def calc_sgn_bkg_cumsums(truth, preds, weights, bins, reverse = True):
    """Calculate signal/background cumulative sum histograms"""
    result = []

    for pred_idx in range(preds.shape[1]):
        rhist_list = get_sgn_bkg_preds(
            truth, preds, weights, pred_idx, pred_idx, bins = bins
        )

        result.append(tuple(rhist1d_cumsum(x, reverse) for x in rhist_list))

    return result

def calc_fom_from_cumsums(rhist_sgn_cumsum, rhist_bkg_cumsum, fom_func):
    """Calculate FOM from signal/background cumulative sum histograms"""
    fom_hist   = fom_func(rhist_sgn_cumsum, rhist_bkg_cumsum)
    fom_err_sq = fom_hist

    return RHist1D(rhist_sgn_cumsum.bins, fom_hist, fom_err_sq)

def calc_foms(sgn_bkg_cumsums, fom_func):
    """Calculate FOMs for a list of signal/background cumulative sum hists"""
    result = []

    for (rhist_sgn_cumsum, rhist_bkg_cumsum) in sgn_bkg_cumsums:
        rhist_fom = calc_fom_from_cumsums(
            rhist_sgn_cumsum, rhist_bkg_cumsum, fom_func
        )

        result.append(rhist_fom)

    return result

