"""Definition of a data resampler that tries to balance targets"""

import logging
import numpy as np

from lstm_ee.data.data_loader.data_slice import DataSlice

LOGGER = logging.getLogger('slice_lid.data.data_loader.balanced_sampler')

class BalancedSampler(DataSlice):
    """
    Decorator around `IDataLoader` that resamples samples to balance targets.

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
        Name of the variable in `data_loader` that indicates whether
        interaction is a charge current interaction.
    pdg_iscc_list : list of (int, bool)
        List of pairs of the form
            [ (pdg_1, iscc_1), (pdg_2, iscc_2), ..., (pdg_N, iscc_N) ]
        that define targets to be balanced.
        Each sample from a `data_loader` is matched against pairs in
        `pdg_iscc_list`.
        If sample pdg and iscc values match some pair (pdg_i, iscc_i) in the
        list `pdg_iscc_list` then `BalancedSampler` will assign target i to
        that sample.
        If sample pdg and iscc values do not match any pair in the
        `pdg_iscc_list` then `BalancedSampler` will assign target 0 to that
        sample.
        The matching of sample pdg and iscc values to the pair of
        (pdg_i, iscc_i) from the `pdg_iscc_list` is done according to the
        following rules:
            1. Sample (pdg, iscc) matches (pdg_i, iscc_i) from `pdg_iscc_list`
               iff (pdg matches pdg_i) and (iscc matches iscc_i).
            2. If pdg_i is None, then all sample pdg values will match it.
            3. If (pdg_i is not None) and `pdg_signed` then the match is
               determined by a result of comparison (pdg == pdg_i).
            4. If (pdg_i is not None) and (not `pdg_signed`) then the match is
               determined by a result of comparison (abs(pdg) == pdg_i).
            5. If iscc_i is None, then all sample iscc values will match it.
            6. If iscc_i is not None, then match is determined by a result
               of evaluation (iscc == iscc_i)
    seed : int or None
        Seed to initialize PRG for random sampling.
    pdg_signed : bool
        If True then `BalancedSampler` will pay attention to the sign of pdg
        values in `pdg_iscc_list` when assigning targets to samples.

    See Also
    --------
    DataFilter
    """

    def __init__(
        self, data_loader, var_pdg, var_iscc, pdg_iscc_list,
        seed, pdg_signed = False
    ):
        self._data_loader   = data_loader
        self._var_pdg       = var_pdg
        self._var_iscc      = var_iscc
        self._pdg_iscc_list = pdg_iscc_list
        self._seed          = seed
        self._pdg_signed    = pdg_signed
        self._mask_dict     = None

        self._calc_target_mask()
        indices = self._calc_resample_index()

        super(BalancedSampler, self).__init__(data_loader, indices)

    def _calc_target_mask(self):
        """Build dict of { target : indices }"""
        mask_dict = { }
        size      = len(self._data_loader)

        selected_mask = np.full(size, False)

        for pdg_value, iscc_value in self._pdg_iscc_list:
            mask = np.full(size, True)

            if pdg_value is not None:
                pdg = self._data_loader.get(self._var_pdg, None)
                if not self._pdg_signed:
                    pdg = np.abs(pdg)

                mask &= (pdg == pdg_value)

            if iscc_value is not None:
                iscc = self._data_loader.get(self._var_iscc, None)
                mask &= (iscc == iscc_value)

            mask_dict[(pdg_value,iscc_value)] = mask
            selected_mask |= mask

        unselected_mask = ~selected_mask

        if np.any(unselected_mask):
            mask_dict[None] = ~selected_mask

        self._mask_dict = mask_dict

    def _calc_resample_index(self):
        """Find slice of a decorate object with equal target representation"""
        index_dict = {
            k : np.nonzero(v)[0] for k,v in self._mask_dict.items()
        }
        LOGGER.debug(
            "BalancedSampler: Target counts: %s",
            { k : len(v) for k,v in index_dict.items() }
        )

        min_target_size = min([ len(v) for v in index_dict.values() ])

        indices = None
        for v in index_dict.values():
            # TODO: use a separate PRG
            np.random.seed(self._seed)
            selected_indices = np.random.choice(
                v, min_target_size, replace = False
            )

            if indices is None:
                indices = selected_indices
            else:
                indices = np.hstack([indices, selected_indices])

        np.random.seed(self._seed)
        np.random.shuffle(indices)

        return indices

