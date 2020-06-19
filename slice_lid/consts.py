"""
Constants that are widely used in slice_lid
"""

import os

DEF_SEED  = 1337
DEF_MASK  = 0.

if 'SLICE_LID_DATADIR' in os.environ:
    ROOT_DATADIR = os.environ['SLICE_LID_DATADIR']
else:
    ROOT_DATADIR = '/'

if 'SLICE_LID_OUTDIR' in os.environ:
    ROOT_OUTDIR = os.environ['SLICE_LID_OUTDIR']
else:
    ROOT_OUTDIR = '/'

