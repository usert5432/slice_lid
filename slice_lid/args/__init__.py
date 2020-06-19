"""
This module contains two classes `Args` and `Config` and a number of functions
to work with them.

`Config` is a structure that holds parameters required to reproduce the
training (e.g. dataset, learning rate, model, etc).

`Args` is a structure that holds an instance of `Config`, plus additional
parameters that are not required to reproduce the training, but useful
nevertheless (e.g. whether to cache dataset when training).
"""

from lstm_ee.args import join_dicts

from .args   import Args
from .config import Config

__all__ = [ 'Args', 'Config', 'join_dicts' ]

