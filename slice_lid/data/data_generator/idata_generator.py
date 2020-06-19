"""
Definition of a DataGenerator interface.
"""

class IDataGenerator:
    """An interface to a DataGenerator.

    An `IDataGenerator` holds an instance of `IDataLoader` object and creates
    batches of data from it.
    """

    def __init__(self):
        self._vars_png3d           = None
        self._vars_slice           = None
        self._var_pdg              = None
        self._var_iscc             = None
        self._target_pdg_iscc_list = None
        self._data_loader          = None
        self._weights              = None

    def get_target_data(self, index):
        """Return an array of targets

        Parameters
        ----------
        index : ndarray, shape (N,) or None
            Index to slice target array. If None, then the entire target array
            will be returned.

        Returns
        -------
        ndarray, shape (M, N_TARGET)
            Array of targets encoded with one-hot encoding.
            N_TARGET = len(self.target_pdg_iscc_list) + 1
            If index is None, then M = len(self.data_loader),
            otherwise M = len(index)
        """
        raise NotImplementedError

    @property
    def target_pdg_iscc_list(self):
        """List of (pdg, iscc) pairs that defined targets of `self`"""
        return self._target_pdg_iscc_list

    @property
    def vars_input_png3d(self):
        """List of 3D prong level input var names to create batches for"""
        return self._vars_png3d

    @property
    def vars_input_slice(self):
        """List of slice level input var names to create batches for"""
        return self._vars_slice

    @property
    def var_target_pdg(self):
        """Name of a variable that holds sample PDG values"""
        return self._var_pdg

    @property
    def var_target_iscc(self):
        """
        Name of a variable that indicates if interaction is Charged Current
        """
        return self._var_iscc

    @property
    def data_loader(self):
        """`IDataLoader` values from which will be used to create batches"""
        return self._data_loader

    @property
    def weights(self):
        """`np.ndarray` of sample weights, shape (len(self.data_loader),)"""
        return self._weights

    def __len__(self):
        """Number of batches this `IDataGenerator` is capable of generating"""
        raise NotImplementedError

    def __getitem__(self, index):
        """Get batch with index `index`.

        Returns a batch constructed from self.data_loader with index `index`.

        Parameters
        ----------
        index : int
            Batch index. 0 <= `index` < len(self)

        Returns
        -------
        inputs : dict
            Dictionary of batches of input variables where keys are input
            labels: [ 'input_slice', 'input_png3d' ] and values are the batches
            themselves.
            If self.vars_input_* is None then the corresponding input will be
            missing from `inputs`.
        targets : dict
            Dictionary of batches of target variables where keys are target
            labels: [ 'target' ] and values are the batches themselves.
        weight : list of ndarray
            List of weights for each target is `targets`.
        """

        raise NotImplementedError

