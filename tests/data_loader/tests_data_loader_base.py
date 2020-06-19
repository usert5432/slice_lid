"""Template functions for DataLoader tests"""

import numpy as np

class FuncsDataLoaderBase():
    """
    Functions to compare data produced by IDataLoader to the expected data
    """
    # pylint: disable=no-member

    def _retrieve_test_null_data(self, data, data_loader, var, index):
        self.assertEqual(len(data_loader), len(data[var]))

        data_test = data_loader.get(var, index)

        if index is None:
            data_null = np.array(data[var])
        else:
            data_null = np.array(data[var])[index]

        self.assertEqual(len(data_test), len(data_null))

        return (data_test, data_null)

    def _compare_scalar_vars(
        self, data, data_loader, var, index = None, sort = False
    ):

        data_test, data_null = self._retrieve_test_null_data(
            data, data_loader, var, index
        )
        if sort:
            data_test = np.sort(data_test)
            data_null = np.sort(data_null)

        self.assertTrue(np.all(np.isclose(data_test, data_null)))


