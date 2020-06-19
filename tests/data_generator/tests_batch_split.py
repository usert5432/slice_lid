"""Tests of correctness of data split into batches by a `DataGenerator`"""

import unittest
import numpy as np

from ..data import X_SLICE_1, X_SLICE_2, X_PNG3D_1, X_PNG3D_2
from .tests_data_generator_base import (
    TestsDataGeneratorBase, make_data_generator
)

TARGET_PDG_ISCC_LIST = [ (0,1), (5,6) ]
TARGETS = [
    [ 0, 1, 0 ],
    [ 1, 0, 0 ],
    [ 0, 0, 1 ],
    [ 0, 1, 0 ],
    [ 1, 0, 0 ],
]

class TestsDataBatchSplit(TestsDataGeneratorBase, unittest.TestCase):
    """Test `DataGenerator` data batching correctness"""

    def test_batch_size_1(self):
        """Test correctness of batch generation with batch_size=1"""
        batch_size = 1
        batch_data = [
            {
                'input_slice'    : [
                    [X_SLICE_1[0], X_SLICE_2[0]],
                ],
                'input_png3d'    : [
                    [
                        [X_PNG3D_1[0][0], X_PNG3D_2[0][0]],
                        [X_PNG3D_1[0][1], X_PNG3D_2[0][1]],
                        [X_PNG3D_1[0][2], X_PNG3D_2[0][2]],
                    ],
                ],
                'target' : [ TARGETS[0] ],
            },
            {
                'input_slice'    : [
                    [X_SLICE_1[1], X_SLICE_2[1]],
                ],
                'input_png3d'    : [
                    [
                        [X_PNG3D_1[1][0], X_PNG3D_2[1][0]],
                    ],
                ],
                'target' : [ TARGETS[1] ],
            },
            {
                'input_slice'    : [
                    [X_SLICE_1[2], X_SLICE_2[2]],
                ],
                'input_png3d'    : np.empty((1, 0, 2)),
                'target' : [ TARGETS[2] ],
            },
            {
                'input_slice'    : [
                    [X_SLICE_1[3], X_SLICE_2[3]],
                ],
                'input_png3d'    : [
                    [
                        [X_PNG3D_1[3][0], X_PNG3D_2[3][0]],
                    ],
                ],
                'target' : [ TARGETS[3] ],
            },
            {
                'input_slice'    : [
                    [X_SLICE_1[4], X_SLICE_2[4]],
                ],
                'input_png3d'    : [
                    [
                        [X_PNG3D_1[4][0], X_PNG3D_2[4][0]],
                        [X_PNG3D_1[4][1], X_PNG3D_2[4][1]],
                    ],
                ],
                'target' : [ TARGETS[4] ],
            },
        ]

        dgen = make_data_generator(
            batch_size           = batch_size,
            target_pdg_iscc_list = TARGET_PDG_ISCC_LIST
        )

        self._compare_dgen_to_batch_data(dgen, batch_data)

    def test_batch_size_2(self):
        """Test correctness of batch generation with batch_size=2"""
        batch_size = 2
        batch_data = [
            {
                'input_slice'    : [
                    [X_SLICE_1[0], X_SLICE_2[0]],
                    [X_SLICE_1[1], X_SLICE_2[1]],
                ],
                'input_png3d'    : [
                    [
                        [X_PNG3D_1[0][0], X_PNG3D_2[0][0]],
                        [X_PNG3D_1[0][1], X_PNG3D_2[0][1]],
                        [X_PNG3D_1[0][2], X_PNG3D_2[0][2]],
                    ],
                    [
                        [X_PNG3D_1[1][0], X_PNG3D_2[1][0]],
                        [np.nan,          np.nan],
                        [np.nan,          np.nan],
                    ],
                ],
                'target' : [ TARGETS[0], TARGETS[1], ],
            },
            {
                'input_slice'    : [
                    [X_SLICE_1[2], X_SLICE_2[2]],
                    [X_SLICE_1[3], X_SLICE_2[3]],
                ],
                'input_png3d'    : [
                    [
                        [np.nan,          np.nan],
                    ],
                    [
                        [X_PNG3D_1[3][0], X_PNG3D_2[3][0]],
                    ],
                ],
                'target' : [ TARGETS[2], TARGETS[3], ],
            },
            {
                'input_slice'    : [
                    [X_SLICE_1[4], X_SLICE_2[4]],
                ],
                'input_png3d'    : [
                    [
                        [X_PNG3D_1[4][0], X_PNG3D_2[4][0]],
                        [X_PNG3D_1[4][1], X_PNG3D_2[4][1]],
                    ],
                ],
                'target' : [ TARGETS[4] ],
            },
        ]

        dgen = make_data_generator(
            batch_size           = batch_size,
            target_pdg_iscc_list = TARGET_PDG_ISCC_LIST
        )
        self._compare_dgen_to_batch_data(dgen, batch_data)

    def test_batch_size_3(self):
        """Test correctness of batch generation with batch_size=3"""
        batch_size = 3
        batch_data = [
            {
                'input_slice'    : [
                    [X_SLICE_1[0], X_SLICE_2[0]],
                    [X_SLICE_1[1], X_SLICE_2[1]],
                    [X_SLICE_1[2], X_SLICE_2[2]],
                ],
                'input_png3d'    : [
                    [
                        [X_PNG3D_1[0][0], X_PNG3D_2[0][0]],
                        [X_PNG3D_1[0][1], X_PNG3D_2[0][1]],
                        [X_PNG3D_1[0][2], X_PNG3D_2[0][2]],
                    ],
                    [
                        [X_PNG3D_1[1][0], X_PNG3D_2[1][0]],
                        [np.nan,          np.nan],
                        [np.nan,          np.nan],
                    ],
                    [
                        [np.nan,          np.nan],
                        [np.nan,          np.nan],
                        [np.nan,          np.nan],
                    ],
                ],
                'target' : [
                    TARGETS[0],
                    TARGETS[1],
                    TARGETS[2],
                ],
            },
            {
                'input_slice'    : [
                    [X_SLICE_1[3], X_SLICE_2[3]],
                    [X_SLICE_1[4], X_SLICE_2[4]],
                ],
                'input_png3d'    : [
                    [
                        [X_PNG3D_1[3][0], X_PNG3D_2[3][0]],
                        [np.nan,          np.nan],
                    ],
                    [
                        [X_PNG3D_1[4][0], X_PNG3D_2[4][0]],
                        [X_PNG3D_1[4][1], X_PNG3D_2[4][1]],
                    ],
                ],
                'target' : [
                    TARGETS[3],
                    TARGETS[4],
                ],
            },
        ]

        dgen = make_data_generator(
            batch_size           = batch_size,
            target_pdg_iscc_list = TARGET_PDG_ISCC_LIST
        )
        self._compare_dgen_to_batch_data(dgen, batch_data)

    def test_batch_size_4(self):
        """Test correctness of batch generation with batch_size=4"""
        batch_size = 4
        batch_data = [
            {
                'input_slice'    : [
                    [X_SLICE_1[0], X_SLICE_2[0]],
                    [X_SLICE_1[1], X_SLICE_2[1]],
                    [X_SLICE_1[2], X_SLICE_2[2]],
                    [X_SLICE_1[3], X_SLICE_2[3]],
                ],
                'input_png3d'    : [
                    [
                        [X_PNG3D_1[0][0], X_PNG3D_2[0][0]],
                        [X_PNG3D_1[0][1], X_PNG3D_2[0][1]],
                        [X_PNG3D_1[0][2], X_PNG3D_2[0][2]],
                    ],
                    [
                        [X_PNG3D_1[1][0], X_PNG3D_2[1][0]],
                        [np.nan,          np.nan],
                        [np.nan,          np.nan],
                    ],
                    [
                        [np.nan,          np.nan],
                        [np.nan,          np.nan],
                        [np.nan,          np.nan],
                    ],
                    [
                        [X_PNG3D_1[3][0], X_PNG3D_2[3][0]],
                        [np.nan,          np.nan],
                        [np.nan,          np.nan],
                    ],
                ],
                'target' : [
                    TARGETS[0],
                    TARGETS[1],
                    TARGETS[2],
                    TARGETS[3],
                ],
            },
            {
                'input_slice'    : [
                    [X_SLICE_1[4], X_SLICE_2[4]],
                ],
                'input_png3d'    : [
                    [
                        [X_PNG3D_1[4][0], X_PNG3D_2[4][0]],
                        [X_PNG3D_1[4][1], X_PNG3D_2[4][1]],
                    ],
                ],
                'target' : [
                    TARGETS[4],
                ],
            },
        ]

        dgen = make_data_generator(
            batch_size           = batch_size,
            target_pdg_iscc_list = TARGET_PDG_ISCC_LIST
        )
        self._compare_dgen_to_batch_data(dgen, batch_data)

    def test_batch_size_5(self):
        """Test correctness of batch generation with batch_size=5 and above"""
        batch_size = 5
        batch_data = [
            {
                'input_slice'    : [
                    [X_SLICE_1[0], X_SLICE_2[0]],
                    [X_SLICE_1[1], X_SLICE_2[1]],
                    [X_SLICE_1[2], X_SLICE_2[2]],
                    [X_SLICE_1[3], X_SLICE_2[3]],
                    [X_SLICE_1[4], X_SLICE_2[4]],
                ],
                'input_png3d'    : [
                    [
                        [X_PNG3D_1[0][0], X_PNG3D_2[0][0]],
                        [X_PNG3D_1[0][1], X_PNG3D_2[0][1]],
                        [X_PNG3D_1[0][2], X_PNG3D_2[0][2]],
                    ],
                    [
                        [X_PNG3D_1[1][0], X_PNG3D_2[1][0]],
                        [np.nan,          np.nan],
                        [np.nan,          np.nan],
                    ],
                    [
                        [np.nan,          np.nan],
                        [np.nan,          np.nan],
                        [np.nan,          np.nan],
                    ],
                    [
                        [X_PNG3D_1[3][0], X_PNG3D_2[3][0]],
                        [np.nan,          np.nan],
                        [np.nan,          np.nan],
                    ],
                    [
                        [X_PNG3D_1[4][0], X_PNG3D_2[4][0]],
                        [X_PNG3D_1[4][1], X_PNG3D_2[4][1]],
                        [np.nan,          np.nan],
                    ],
                ],
                'target' : TARGETS,
            },
        ]

        dgen = make_data_generator(
            batch_size           = batch_size,
            target_pdg_iscc_list = TARGET_PDG_ISCC_LIST
        )
        self._compare_dgen_to_batch_data(dgen, batch_data)

        dgen = make_data_generator(
            batch_size           = batch_size + 1,
            target_pdg_iscc_list = TARGET_PDG_ISCC_LIST
        )
        self._compare_dgen_to_batch_data(dgen, batch_data)

        dgen = make_data_generator(
            batch_size           = batch_size + 1024,
            target_pdg_iscc_list = TARGET_PDG_ISCC_LIST
        )
        self._compare_dgen_to_batch_data(dgen, batch_data)

if __name__ == '__main__':
    unittest.main()

