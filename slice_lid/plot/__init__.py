"""
This module contains functions to make various plots.

Notes
-----
During the module initialization it sets the default plot parameters.
It also recognizes presence of the environment variable `BORING_STYLE`.

If `BORING_STYLE` environment variable is set, then the plots style will be
adjusted to better suite article style.
Otherwise, the plot style will be adjusted to look nice in presentations.
"""

import os

import matplotlib.style
import matplotlib as mpl

from cycler import cycler

if 'BORING_STYLE' in os.environ:
    mpl.rcParams['axes.prop_cycle'] = cycler(color = 'krgbcmy')
    mpl.rcParams['font.size']        = 12
    mpl.rcParams['legend.fontsize']  = 'medium'
    mpl.rcParams['xtick.labelsize']  = 10
    mpl.rcParams['ytick.labelsize']  = 10
    mpl.rcParams['figure.titlesize'] = 'medium'

