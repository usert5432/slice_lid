"""
Common dataset definition for various tests.
"""

import numpy as np

def nan_equal(a, b):
    """Compare `a` and `b` taking into account possible NaN values"""
    a = np.array(a)
    b = np.array(b)
    return (np.isclose(a, b) | (np.isnan(a) & np.isnan(b))).all()

TEST_INPUT_VARS_SLICE = [ 'x_slice1', 'x_slice2' ]
TEST_INPUT_VARS_PNG3D = [ 'x_png3d1', 'x_png3d2' ]
TEST_TARGET_VAR_PDG   = 'target_pdg'
TEST_TARGET_VAR_ISCC  = 'target_iscc'

X_SLICE_1 = [1, 2, 3, 2, 1]
X_SLICE_2 = [2, 3, 4, 3, 2]

TARGET_PDG  = [ 0, 0, 5, 0, 3 ]
TARGET_ISCC = [ 1, 0, 6, 1, 3 ]

X_PNG3D_1 = [ [ 1,2,3 ], [ 4 ], [ ], [ 4 ], [ 1,2 ] ]
X_PNG3D_2 = [ [ 4,5,6 ], [ 5 ], [ ], [ 5 ], [ 4,5 ] ]

TEST_DATA = {
    'x_slice1'    : X_SLICE_1,
    'x_slice2'    : X_SLICE_2,
    'target_pdg'  : TARGET_PDG,
    'target_iscc' : TARGET_ISCC,
    'x_png3d1'    : X_PNG3D_1,
    'x_png3d2'    : X_PNG3D_2,
}

TEST_DATA_VARS = list(TEST_DATA.keys())
TEST_DATA_LEN  = len(TEST_DATA['x_slice1'])

