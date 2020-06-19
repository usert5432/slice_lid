"""Helper blocks to create IDataGenerator tests"""

import numpy as np

from lstm_ee.data.data_loader.dict_loader         import DictLoader
from slice_lid.data.data_generator.data_generator import DataGenerator
from ..data import (
    TEST_DATA, TEST_INPUT_VARS_SLICE, TEST_INPUT_VARS_PNG3D,
    TEST_TARGET_VAR_PDG, TEST_TARGET_VAR_ISCC, nan_equal
)

def make_data_generator(
    data_loader      = DictLoader(TEST_DATA),
    vars_input_slice = TEST_INPUT_VARS_SLICE,
    vars_input_png3d = TEST_INPUT_VARS_PNG3D,
    var_target_pdg   = TEST_TARGET_VAR_PDG,
    var_target_iscc  = TEST_TARGET_VAR_ISCC,
    **kwargs
):
    """Create simple `DataGenerator`"""
    # pylint: disable=dangerous-default-value

    return DataGenerator(
        data_loader      = data_loader,
        vars_input_slice = vars_input_slice,
        vars_input_png3d = vars_input_png3d,
        var_target_pdg   = var_target_pdg,
        var_target_iscc  = var_target_iscc,
        **kwargs,
    )

class TestsDataGeneratorBase():
    """Functions to compare generated data batches to the expected ones"""
    # pylint: disable=no-member

    def _compare_np_arrays(self, label, batch_idx, test, null):
        self.assertEqual(
            (label in test), (label in null),
            "Label: %s. Batch Index: %d. Label missing." % (label, batch_idx)
        )

        if label not in test:
            return

        test = np.array(test[label])
        null = np.array(null[label])

        self.assertEqual(
            test.shape, null.shape,
            "Label: %s. Batch Index: %d. Shapes differ" % (label, batch_idx)
        )

        self.assertTrue(
            nan_equal(test, null),
            "Label: %s. Batch Index: %d. Values differ: %s != %s" % (
                label, batch_idx, test, null
            )
        )

    def _compare_dgen_to_batch_data(self, dgen, batch_data):
        self.assertEqual(len(dgen), len(batch_data))

        # pylint: disable=consider-using-enumerate
        for i in range(len(dgen)):
            inputs_test, targets_test = dgen[i][:2]
            null = batch_data[i]

            self._compare_np_arrays('input_slice', i, inputs_test,  null)
            self._compare_np_arrays('input_png3d', i, inputs_test,  null)
            self._compare_np_arrays('target',      i, targets_test, null)

    def _compare_dgen_to_batch_weights(self, dgen, batch_weights):
        self.assertEqual(len(dgen), len(batch_weights))

        # pylint: disable=consider-using-enumerate
        for i in range(len(dgen)):
            weights_test = { 'weight' : dgen[i][2][0] }
            weights_null = { 'weight' : batch_weights[i] }

            self._compare_np_arrays('weight', i, weights_test, weights_null)

