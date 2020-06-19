"""
Test runner to run all available tests.

Examples
--------
Run all available tests:

$ python -m unittest tests.run_tests.suite
"""

import unittest

import tests.data_loader.tests_balanced_sampler
import tests.data_loader.tests_data_filter

import tests.data_generator.tests_batch_split
import tests.data_generator.tests_class_weights_calc
import tests.data_generator.tests_class_weights

def suite():
    """Construct test suite"""
    result = unittest.TestSuite()
    loader = unittest.TestLoader()

    result.addTest(loader.loadTestsFromModule(
        tests.data_loader.tests_balanced_sampler
    ))
    result.addTest(loader.loadTestsFromModule(
        tests.data_loader.tests_data_filter
    ))
    result.addTest(loader.loadTestsFromModule(
        tests.data_generator.tests_batch_split
    ))
    result.addTest(loader.loadTestsFromModule(
        tests.data_generator.tests_class_weights_calc
    ))
    result.addTest(loader.loadTestsFromModule(
        tests.data_generator.tests_class_weights
    ))

    return result

def run():
    """Run test suite"""
    runner = unittest.TextTestRunner(verbosity = 3)
    runner.run(suite())

if __name__ == '__main__':
    run()

