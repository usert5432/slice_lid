"""Test `IDataLoader` data filtering by a `DataFilter` decorator"""

import unittest

from lstm_ee.data.data_loader.dict_loader   import DictLoader
from slice_lid.data.data_loader.data_filter import DataFilter

from .tests_data_loader_base import FuncsDataLoaderBase

class TestsDataFilter(unittest.TestCase, FuncsDataLoaderBase):
    """Test `DataFilter` decorator"""

    def test_filter_simple(self):
        """Simple filtering tests"""
        data = {
            'pdg'  : [ 1, 2, 0, 1, 2 ],
            'iscc' : [ 0, 1, 0, 1, 0 ],
            'idx'  : [ 0, 1, 2, 3, 4 ],
        }
        keep_pdg_iscc_list = [ (0, 0), (1, 0) ]
        slice_data  = { 'idx' : [ 0, 2 ] }
        data_loader = DataFilter(
            DictLoader(data), 'pdg', 'iscc', keep_pdg_iscc_list
        )

        self._compare_scalar_vars(slice_data, data_loader, 'idx')

    def test_filter_missing_value(self):
        """Test filtering with filter that does not match anything"""
        data = {
            'pdg'  : [ 1, 2, 0, 1, 2 ],
            'iscc' : [ 0, 1, 0, 1, 0 ],
            'idx'  : [ 0, 1, 2, 3, 4 ],
        }
        keep_pdg_iscc_list = [ (0, 0), (-1, 0) ]
        slice_data  = { 'idx' : [ 2 ] }
        data_loader = DataFilter(
            DictLoader(data), 'pdg', 'iscc', keep_pdg_iscc_list
        )

        self._compare_scalar_vars(slice_data, data_loader, 'idx')

    def test_filter_pass_all(self):
        """Test filtering that should not filter anything"""
        data = {
            'pdg'  : [ 1, 2, 0, 1, 2 ],
            'iscc' : [ 0, 1, 0, 1, 0 ],
            'idx'  : [ 0, 1, 2, 3, 4 ],
        }
        keep_pdg_iscc_list = [ (0, 0), (1, 0), (1, 1), (2, 0), (2, 1) ]
        slice_data  = { 'idx' : [ 0, 1, 2, 3, 4 ] }
        data_loader = DataFilter(
            DictLoader(data), 'pdg', 'iscc', keep_pdg_iscc_list
        )

        self._compare_scalar_vars(slice_data, data_loader, 'idx')

    def test_filter_pass_none(self):
        """Test filtering that should reject all samples"""
        data = {
            'pdg'  : [ 1, 2, 0, 1, 2 ],
            'iscc' : [ 0, 1, 0, 1, 0 ],
            'idx'  : [ 0, 1, 2, 3, 4 ],
        }
        keep_pdg_iscc_list = [ (-1, -1) ]
        slice_data  = { 'idx' : [ ] }
        data_loader = DataFilter(
            DictLoader(data), 'pdg', 'iscc', keep_pdg_iscc_list
        )

        self._compare_scalar_vars(slice_data, data_loader, 'idx')

    def test_filter_wildcard_pdg(self):
        """Test filtering with wildcard PDG pattern"""
        data = {
            'pdg'  : [ 1, 2, 0, 1, 2 ],
            'iscc' : [ 0, 1, 0, 1, 0 ],
            'idx'  : [ 0, 1, 2, 3, 4 ],
        }
        keep_pdg_iscc_list = [ (None, 1) ]
        slice_data  = { 'idx' : [ 1, 3 ] }
        data_loader = DataFilter(
            DictLoader(data), 'pdg', 'iscc', keep_pdg_iscc_list
        )

        self._compare_scalar_vars(slice_data, data_loader, 'idx')

    def test_filter_wildcard_iscc(self):
        """Test filtering with wildcard ISCC pattern"""
        data = {
            'pdg'  : [ 1, 2, 0, 1, 2 ],
            'iscc' : [ 0, 1, 0, 1, 0 ],
            'idx'  : [ 0, 1, 2, 3, 4 ],
        }
        keep_pdg_iscc_list = [ (1, None) ]
        slice_data  = { 'idx' : [ 0, 3 ] }
        data_loader = DataFilter(
            DictLoader(data), 'pdg', 'iscc', keep_pdg_iscc_list
        )

        self._compare_scalar_vars(slice_data, data_loader, 'idx')

if __name__ == '__main__':
    unittest.main()

