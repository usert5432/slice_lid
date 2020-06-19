"""Definition of a transformation that filters samples based on target"""

import logging
import numpy as np

from lstm_ee.data.data_loader.data_slice import DataSlice

LOGGER = logging.getLogger('slice_lid.data.data_loader.data_filter')

class DataFilter(DataSlice):
    """Decorator around `IDataLoader` that filters samples based on targets.

    `BalancedSampler` randomly selects subsample from the decorated object in
    order to make sure that the targets are represented equally in a subsample.
    The non-selected part from the decorated object is discarded.

    Parameters
    ----------
    data_loader : IDataLoader
        DataLoader to decorate.
    var_pdg : str
        Name of the variable in `data_loader` that holds particle PDG values.
    var_iscc : str
        Name of the variable in `data_loader` that indicates whether interaction
        is a charge current interaction.
    keep_pdg_iscc_list : list of (int, bool)
        List of pairs of the form
            [ (pdg_1, iscc_1), (pdg_2, iscc_2), ..., (pdg_N, iscc_N) ]
        that define targets to be kept.
        Each sample from a `data_loader` is matched against pairs in
        `keep_pdg_iscc_list`.
        If sample pdg and iscc values match some pair (pdg_i, iscc_i) in the
        list `keep_pdg_iscc_list` then `DataFilter` will keep that sample.
        If sample pdg and iscc values do not match any pair in the
        `keep_pdg_iscc_list` then `DataFilter` will drop that sample.
        The matching of sample pdg and iscc values to the pair of
        (pdg_i, iscc_i) from the `keep_pdg_iscc_list` is done according to the
        following rules:
            1. Sample (pdg, iscc) matches (pdg_i, iscc_i) from
               `keep_pdg_iscc_list` iff
               (pdg matches pdg_i) and (iscc matches iscc_i).
            2. If pdg_i is None, then all sample pdg values will match it.
            3. If (pdg_i is not None) then match determined by a result of
               comparison (pdg == pdg_i).
            4. If iscc_i is None, then all sample iscc values will match it.
            5. If iscc_i is not None, then match is determined by a result of
               comparison (iscc == iscc_i)

    See Also
    --------
    BalancedSampler
    """

    def __init__(
        self, data_loader, var_pdg, var_iscc, keep_pdg_iscc_list
    ):
        self._data_loader = data_loader
        self._var_pdg     = var_pdg
        self._var_iscc    = var_iscc
        self._keep_list   = keep_pdg_iscc_list

        indices = self._calc_filtered_index()

        super(DataFilter, self).__init__(data_loader, indices)

    def _calc_filtered_index(self):
        """Find slice to be kept"""
        size = len(self._data_loader)
        filter_mask = np.full(size, False)

        for pdg_value, iscc_value in self._keep_list:
            mask = np.full(size, True)

            if pdg_value is not None:
                pdg = self._data_loader.get(self._var_pdg, None)
                mask &= (pdg == pdg_value)

            if iscc_value is not None:
                iscc = self._data_loader.get(self._var_iscc, None)
                mask &= (iscc == iscc_value)

            filter_mask |= mask

        indices = np.nonzero(filter_mask)[0]
        LOGGER.debug(
            "DataFilter: %d entries pass filter out of %d (%.2f %%)",
            len(indices), size,
            0 if (size == 0) else 100 * len(indices) / size
        )

        return indices

