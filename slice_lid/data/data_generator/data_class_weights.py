"""
A definition of a decorator that modifies weights in a batch.
"""

import logging

from .idata_decorator import IDataDecorator
from .weights.class_weights import calc_class_weights

LOGGER = logging.getLogger('slice_lid.data.dgen')

class DataClassWeights(IDataDecorator):
    """A decorator around `IDataGenerator` that modifies target weights.

    This decorator will calculate global class weights and translate them into
    sample weights for each sample. Then weight returned by a decorated object
    will be multiplicatively modified by the calculated sample weights.

    Parameters
    ----------
    dgen : IDataGenerator
        `IDataGenerator` to be decorated.
    class_weights :  { 'equal', None }, optional
        Name of the class weights to be calculated (c.f. `calc_class_weights`).
        If None, then the data returned by a decorated object will not be
        modified. Default: None.

    See Also
    --------
    calc_class_weights
    """

    def __init__(self, dgen, class_weights = None):
        super(DataClassWeights, self).__init__(dgen)

        self._class_weights = class_weights
        self._init_class_weights()

    def _init_class_weights(self):
        """Calculate class weights"""
        if self._class_weights is None:
            return

        LOGGER.info("Calculating class weights...")

        targets             = self.get_target_data(None)
        self._class_weights = calc_class_weights(self._class_weights, targets)
        self._weights       = self._calc_weights(targets)

        LOGGER.debug("Class weights are: %s", self._class_weights)

    def _calc_weights(self, targets):
        """Calculate sample weights from class weights"""

        if self._class_weights is None:
            return None

        class_indices  = targets.argmax(axis = 1).ravel()
        sample_weights = self._class_weights[class_indices]

        return sample_weights

    def __getitem__(self, index):

        if self._class_weights is None:
            return self._dgen[index]

        inputs, targets, weights = self._dgen[index]
        sample_weights = self._calc_weights(targets['target'])

        return (inputs, targets, [x * sample_weights for x in weights ])

