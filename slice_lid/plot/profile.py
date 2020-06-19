"""
Functions to make plots of error matrix components vs training parameter(-s).
"""

from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np

from cafplot.plot import save_fig
from lstm_ee.plot.profile import get_x_scales, prepare_x_var
from .labels import convert_targets_to_labels

def plot_profile_base(x, y_dict, label_x, label_y, categorical, scale_x):
    """Make a plot of error matrix components `y_dict` vs training param `x`"""
    f, ax = plt.subplots()

    if scale_x is not None:
        ax.set_xscale(scale_x)

    keys = sorted(list(y_dict.keys()))

    for idx,k in enumerate(keys):
        ax.plot(x, y_dict[k], marker = 'o', color = 'C%d' % (idx), label = k)

    ax.set_title('%s vs %s' % (label_y, label_x))

    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)

    if categorical:
        x = [ "\n".join(wrap(l, 70)) for l in x ]
        ax.set_xticklabels(x)
        f.autofmt_xdate(rotation = 45)

    ax.legend()

    return f, ax

def sort_data(x, y_dict, sort_type, y_key):
    """
    Simultaneously sort (x, y_dict) by values from `y_dict` with key `y_key`
    """
    if sort_type is None:
        return (x, y_dict)

    if sort_type == 'x':
        sorted_indices = np.argsort(x)
    elif sort_type == 'y':
        sorted_indices = np.argsort(y_dict[y_key])
    else:
        raise ValueError("Unknown sort type")

    return (
        x[sorted_indices], { k : v[sorted_indices] for k,v in y_dict.items() },
    )

def get_average_accuracy(err_mat):
    """Get average of diagonal components of the error matrix"""
    return np.nanmean(np.diag(err_mat))

def plot_profile(
    var_list, err_mat_list, target_pdg_iscc_list, label_x, label_y,
    sort_type   = None,
    categorical = True,
    fname       = None,
    ext         = 'png'
):
    """
    Make and save plots of error matrix components vs training parameter.

    This function will make and save a number plots (one for different
    axis scales) of the diagonal components of the error matrices in
    the `err_mat_list` vs values of `var_list`.

    Parameters
    ----------
    var_list : list
        List of values of the configuration parameters. Each element in
        `var_list` specifies a different training. Value of the configuration
        parameter can be of any type. These values will be used for the x axis.
    err_mat_list : list of ndarray
        List of error matrices. Once for each element in `var_list`.
    target_pdg_iscc_list : list of (int, bool)
        List of target specifications. C.f. `slice_lid.Config`.
    label_y : str or None
        Label of the x axis.
    label_y : str or None
        Label of the y axis.
    sort_type : { 'x', 'y', None }, optional
        If not None, the the points will be ordered by their coordinate
        specified by `sort_type`.
        For example, if the configuration parameter is a categorical variable
        (e.g. model name) then the x axis won't have any natural order, and
        it may make sense to order points by their y coordinates.
        Default: None.
    categorical : bool, optional
        Whether to assume that the x variable is categorical (as opposed to
        numerical). For example, if `var_list` contains values of the learning
        rate, then it is a numerical variable. On the other hand if `var_list`
        contains names of the models (str), then such variable cannot be
        represented as a number and therefore categorical.
        If x variable is categorical then it does not make sense to plot it
        in logarithmic scale or convert values to numbers. `categorical`
        parameter hints `plot_profile_base` not to do those things.
        Default: True
    fname : str
        Prefix of the path that will be used to build plot file names.
    ext : str or list of str
        Extension of the plots. If list then the plots will be saved in
        multiple formats.
    """
    x      = prepare_x_var(var_list, categorical)
    labels = convert_targets_to_labels(target_pdg_iscc_list)

    y_dict = {
        k : np.array([ y[idx][idx] for y in err_mat_list ]) \
            for idx,k in enumerate(labels)
    }
    y_dict['Average'] = np.array(
        [ get_average_accuracy(y) for y in err_mat_list ]
    )

    x, y_dict    = sort_data(x, y_dict, sort_type, 'Average')
    scale_x_list = get_x_scales(x, categorical)

    for scale_x in scale_x_list:
        f, _ = plot_profile_base(
            x, y_dict, label_x, label_y, categorical, scale_x
        )

        fullname = "%s_xs(%s)" % (fname, scale_x)

        save_fig(f, fullname, ext)
        plt.close(f)

