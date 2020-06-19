"""
Definition of a DataGenerator that creates data batches from DataLoader
"""

import math
import numpy as np

from lstm_ee.data.data_generator.funcs.funcs_varr import unpack_varr_arrays

from .idata_generator import IDataGenerator

class DataGenerator(IDataGenerator):
    """Primary `slice_lid` DataGenerator that batches data from a `IDataLoader`

    This generator takes an instance of `IDataLoader` and specification of
    input/target variables and creates batches of data based on them.
    Batches can be retrieved with __getitem__ method that takes batch
    index as an input.

    Parameters
    ----------
    data_loader : `IDataLoader`
        `IDataLoader` which will be used to retrieve values of variables.
    batch_size : int
        Size of the batches to be generated.
    max_prongs : int or None, optional
        If `max_prongs` is not None, then the number of 3D prongs will be
        truncated by `max_prongs`. Default: None.
    target_pdg_iscc_list : list of (int, bool)
        This parameters specifies target classes of the `slice_lid` network.
        The specification has the form
            [ (pdg_1, iscc_1), (pdg_2, iscc_2), ... (pdg_N, iscc_N) ]
        C.f. `slice_lid.args.Config.target_pdg_iscc_list`
    vars_input_slice : list of str or None, optional
        Names of slice level input variables in `data_loader`.
        If None no slice level inputs will be generated. Default: None.
    vars_input_png3d : list of str or None, optional
        Names of 3d prong level input variables in `data_loader`.
        If None no 3d prong level inputs will be generated. Default: None.
    var_target_pdg : str
        Name of the variable that has true PDG number of the event.
    var_target_iscc : str
        Name of the variable that specifies whether given event was a Charged
        Current event.

    See Also
    --------
    slice_lid.args.Config : for detailed parameter explanations.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        data_loader,
        batch_size,
        max_prongs           = None,
        target_pdg_iscc_list = None,
        vars_input_slice     = None,
        vars_input_png3d     = None,
        var_target_pdg       = None,
        var_target_iscc      = None,
    ):
        super(DataGenerator, self).__init__()

        self._data_loader  = data_loader
        self._batch_size   = batch_size
        self._max_prongs   = max_prongs

        self._target_pdg_iscc_list = target_pdg_iscc_list
        self._vars_input_png3d     = vars_input_png3d
        self._vars_input_slice     = vars_input_slice
        self._var_target_pdg       = var_target_pdg
        self._var_target_iscc      = var_target_iscc

        self._weights = np.ones(len(self._data_loader))

    def __len__(self):
        return math.ceil(len(self._data_loader) / self._batch_size)

    def get_scalar_data(self, variables, index):
        """Generate batch of scalar data.

        Parameters
        ----------
        variables : list of str
            List of variables names which values will be joined into a batch.
        index : int or list of int or None
            Index that defines slice of values to be used when generating
            batch. If None, all available values will be joined into a batch.

        Returns
        -------
        ndarray, shape (N_SAMPLE, len(variables))
            Values of `variables` with sliced by `index` batched together.
        """
        result = np.empty((len(index), len(variables)), dtype = np.float32)

        for idx,vname in enumerate(variables):
            result[:, idx] = self._data_loader.get(vname, index)

        return result

    def get_varr_data(self, variables, index, max_prongs = None):
        """Generate batch of variable length arrays (prong) data.

        All variables length arrays will be batches together into a fixed
        size `np.ndarray`. Missing variable length values will be padded
        by `np.nan`

        Parameters
        ----------
        variables : list of str
            List of variables names which values will be joined into a batch.
        index : int or list of int or None
            Index that defines slice of values to be used when generating
            batch. If None, all available values will be joined into a batch.
        max_prongs : int or None, optional
            If `max_prongs` is not None, then the variable length dimension
            will be truncated by `max_prongs`.

        Returns
        -------
        ndarray, shape (N_SAMPLE, N_VARR, len(variables))
            Values of `variables` with sliced by `index` and batched together.
            Second dimension goes along the variable length axis.

        See Also
        --------
        unpack_varr_arrays
        """
        result = unpack_varr_arrays(
            self._data_loader, variables, index, max_prongs
        )

        return result

    def get_target_data(self, index):
        """Generate batch of targets according to self.target_pdg_iscc_list

        Parameters
        ----------
        index : int or list of int or None
            Index that defines slice of values to be used when generating
            batch. If None, all available values will be joined into a batch.

        Returns
        -------
        ndarray, shape (N_SAMPLE, N_TARGET)
            Batch of targets. N_TARGET = len(self.target_pdg_iscc_list) + 1
        """

        pdg_list  = abs(self._data_loader.get(self._var_target_pdg,  index))
        iscc_list =     self._data_loader.get(self._var_target_iscc, index)

        targets = np.zeros(
            (len(pdg_list), len(self._target_pdg_iscc_list) + 1)
        )

        for idx,(pdg,iscc) in enumerate(self._target_pdg_iscc_list):
            targets[(pdg_list == pdg) & (iscc_list == iscc), idx + 1] = 1

        maxvals = targets.max(1)
        targets[maxvals == 0, 0] = 1

        return targets

    def get_data(self, index):
        """Generate batch of inputs and targets.

        Only variables from self.vars_input_* will be used to generate input
        batches. Batches of target variables will be constructed according
        to `self.target_pdg_iscc_list`.

        Parameters
        ----------
        index : int or list of int or None
            Index of the `IDataLoader` this generator holds that specifies
            slice of values to be batched together.
            If None, all available values will be batched.

        Returns
        -------
        (inputs, targets)
            Dictionaries of input and target batches.

        See Also
        --------
        DataGenerator.__getitem__
        """

        inputs  = {}
        targets = {}

        if self._vars_input_slice is not None:
            inputs['input_slice'] = self.get_scalar_data(
                self._vars_input_slice, index
            )

        if self._vars_input_png3d is not None:
            inputs['input_png3d'] = self.get_varr_data(
                self._vars_input_png3d, index, self._max_prongs
            )

        targets['target'] = self.get_target_data(index)

        return (inputs, targets)

    def __getitem__(self, index):

        start = index * self._batch_size
        end   = min((index + 1) * self._batch_size, len(self._data_loader))
        index = np.arange(start, end)

        batch_data    = self.get_data(index)
        batch_weights = self._weights[start:end]

        return batch_data + ( [batch_weights], )

