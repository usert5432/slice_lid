"""
Test correctness of the class weights calculated by the `DataClassWeights`
"""

import unittest

from lstm_ee.data.data_loader import DictLoader

from slice_lid.data.data_generator.data_generator import DataGenerator
from slice_lid.data.data_generator.data_class_weights import DataClassWeights
from slice_lid.data.data_generator.weights.class_weights import (
    equal_class_weights
)

from .tests_data_generator_base import TestsDataGeneratorBase

class TestClassWeights(TestsDataGeneratorBase, unittest.TestCase):
    """Test `DataClassWeights` decorator"""

    @staticmethod
    def _make_data_generator(pdg, iscc, target_pdg_iscc_list, **kwargs):

        return DataGenerator(
            data_loader          = DictLoader({ 'pdg' : pdg, 'iscc' : iscc }),
            target_pdg_iscc_list = target_pdg_iscc_list,
            vars_input_slice     = None,
            vars_input_png3d     = None,
            var_target_pdg       = 'pdg',
            var_target_iscc      = 'iscc',
            **kwargs,
        )


    def test_equal_weights_batch_size_1(self):
        """Test equal class weights for batch_size=1"""
        batch_size = 1
        pdg_list   = [ 1, 2, 3, 3, 2, 3 ]
        iscc_list  = [ 1, 0, 1, 0, 0, 0 ]

        target_pdg_iscc_list = [ (1, 1), (2, 0) ]

        dgen = TestClassWeights._make_data_generator(
            pdg_list, iscc_list, target_pdg_iscc_list, batch_size = batch_size
        )

        class_weights  = equal_class_weights(dgen.get_target_data(None))
        sample_weights = [
            class_weights[1],
            class_weights[2],
            class_weights[0],
            class_weights[0],
            class_weights[2],
            class_weights[0],
        ]

        batch_weights = [
            [ sample_weights[0] ],
            [ sample_weights[1] ],
            [ sample_weights[2] ],
            [ sample_weights[3] ],
            [ sample_weights[4] ],
            [ sample_weights[5] ],
        ]

        self._compare_dgen_to_batch_weights(
            DataClassWeights(dgen, 'equal'), batch_weights
        )

    def test_equal_weights_batch_size_3(self):
        """Test equal class weights for batch_size=3"""
        batch_size = 3
        pdg_list   = [ 1, 2, 3, 3, 2, 3 ]
        iscc_list  = [ 1, 0, 1, 0, 0, 0 ]

        target_pdg_iscc_list = [ (1, 1), (2, 0) ]

        dgen = TestClassWeights._make_data_generator(
            pdg_list, iscc_list, target_pdg_iscc_list, batch_size = batch_size
        )

        class_weights  = equal_class_weights(dgen.get_target_data(None))
        sample_weights = [
            class_weights[1],
            class_weights[2],
            class_weights[0],
            class_weights[0],
            class_weights[2],
            class_weights[0],
        ]

        batch_weights = [
            [ sample_weights[0], sample_weights[1], sample_weights[2] ],
            [ sample_weights[3], sample_weights[4], sample_weights[5] ],
        ]

        self._compare_dgen_to_batch_weights(
            DataClassWeights(dgen, 'equal'), batch_weights
        )


if __name__ == '__main__':
    unittest.main()


