"""Test `IDataLoader` data balanced by a `BalancedSampler` decorator"""

import unittest
import numpy as np

from lstm_ee.data.data_loader.dict_loader        import DictLoader
from slice_lid.data.data_loader.balanced_sampler import BalancedSampler

from .tests_data_loader_base import FuncsDataLoaderBase

class TestsBalancedSampler(unittest.TestCase, FuncsDataLoaderBase):
    """Test `BalancedSampler` decorator"""

    @staticmethod
    def make_balanced_sampler(data, pdg_iscc_list, seed, pdg_signed = False):
        # pylint: disable=unused-argument
        """Construct simple `BalancedSampler` from dict data"""
        return BalancedSampler(
            DictLoader(data), 'pdg', 'iscc', pdg_iscc_list, seed
        )

    def test_sampler_simple(self):
        """Test simple balancer"""
        seed = 0
        data = {
            'pdg'  : [ 1, 2, 0, 1, 2 ],
            'iscc' : [ 0, 1, 0, 1, 0 ],
            'idx'  : [ 0, 1, 2, 3, 4 ],
        }
        target_pdg_iscc_list = [ (0, 0), (1, 0) ]

        np.random.seed(seed)
        slice_data  = {
            'idx' : [
                0, 2,
                np.random.choice([1, 3, 4], size = 1, replace = False)[0]
            ]
        }
        data_loader = TestsBalancedSampler.make_balanced_sampler(
            data, target_pdg_iscc_list, seed
        )

        self._compare_scalar_vars(slice_data, data_loader, 'idx', sort = True)

    def test_sampler_trivial(self):
        """Test balancer in a case when data has just one target"""
        seed = 0
        data = {
            'pdg'  : [ 1, 2, 0, 1, 2 ],
            'iscc' : [ 0, 1, 0, 1, 0 ],
            'idx'  : [ 0, 1, 2, 3, 4 ],
        }
        target_pdg_iscc_list = [ ]

        slice_data  = { 'idx' : [ 0, 1, 2, 3, 4 ] }
        data_loader = TestsBalancedSampler.make_balanced_sampler(
            data, target_pdg_iscc_list, seed
        )

        self._compare_scalar_vars(slice_data, data_loader, 'idx', sort = True)

    def test_sampler_fully_separate_classes(self):
        """Test balancer in a case when targets are already balanced"""
        seed = 0
        data = {
            'pdg'  : [ 1, 2, 0, 1, 2 ],
            'iscc' : [ 0, 1, 0, 1, 0 ],
            'idx'  : [ 0, 1, 2, 3, 4 ],
        }
        target_pdg_iscc_list = [ (1, 0), (2, 1), (0, 0), (1, 1), (2, 0) ]

        slice_data  = { 'idx' : [ 0, 1, 2, 3, 4 ] }
        data_loader = TestsBalancedSampler.make_balanced_sampler(
            data, target_pdg_iscc_list, seed
        )

        self._compare_scalar_vars(slice_data, data_loader, 'idx', sort = True)


if __name__ == '__main__':
    unittest.main()

