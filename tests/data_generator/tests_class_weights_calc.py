"""Test correctness of the class weights calculations"""

import unittest

from lstm_ee.data.data_loader.dict_loader         import DictLoader
from slice_lid.data.data_generator.data_generator import DataGenerator

from slice_lid.data.data_generator.weights.class_weights import (
    equal_class_weights
)

from ..data import nan_equal

TARGET_PDG_ISCC_LIST = [ (0,1), (5,6) ]
TARGETS = [
    [ 0, 1, 0 ],
    [ 1, 0, 0 ],
    [ 0, 0, 1 ],
    [ 0, 1, 0 ],
    [ 1, 0, 0 ],
]

class TestsClassWeightsCalc(unittest.TestCase):
    """Test correctness of the class weights calculations"""

    @staticmethod
    def _make_dgen(pdg_list, iscc_list, target_pdg_iscc_list):

        data_loader = DictLoader(
            { 'pdg' : pdg_list, 'iscc' : iscc_list }
        )

        return DataGenerator(
            data_loader, 10,
            target_pdg_iscc_list = target_pdg_iscc_list,
            var_target_pdg       = 'pdg',
            var_target_iscc      = 'iscc',
        )

    def test_equal_weights(self):
        """Test equal weight calculation"""

        pdg_list  = [ 1, 2, 3, 3, 2, 3 ]
        iscc_list = [ 1, 0, 1, 0, 0, 0 ]

        target_pdg_iscc_list = [ (1, 1), (2, 0) ]

        # Targets = [
        #   [ 0, 1, 0],
        #   [ 0, 0, 1],
        #   [ 1, 0, 0],
        #   [ 1, 0, 0],
        #   [ 0, 0, 1],
        #   [ 1, 0, 0],
        # ]

        # Counts = [ 3, 1, 2 ]
        # Weights = N / (n_classes * counts) = 6 / (3 * counts)
        # Weights = [ 2 / 3, 2 / 1, 2 / 2 ] = [ 2/3, 2, 1 ]

        null_class_weights = [ 2/3, 2, 1]
        test_class_weights = equal_class_weights(
            TestsClassWeightsCalc._make_dgen(
                pdg_list, iscc_list, target_pdg_iscc_list
            ).get_target_data(None)
        )

        self.assertTrue(nan_equal(test_class_weights, null_class_weights))

if __name__ == '__main__':
    unittest.main()

