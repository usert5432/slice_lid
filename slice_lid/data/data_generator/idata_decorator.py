"""
Definition of an interface of a decorator around `IDataGenerator`.
"""

from .idata_generator import IDataGenerator

class IDataDecorator(IDataGenerator):
    """An interface to a decorator around `IDataGenerator`"""

    def __init__(self, dgen):
        super(IDataDecorator, self).__init__()
        self._dgen = dgen

    def get_target_data(self, index):
        return self._dgen.get_target_data(index)

    @property
    def target_pdg_iscc_list(self):
        return self._dgen.target_pdg_iscc_list

    @property
    def vars_input_png3d(self):
        return self._dgen.vars_input_png3d

    @property
    def vars_input_slice(self):
        return self._dgen.vars_input_slice

    @property
    def var_target_pdg(self):
        return self._dgen.var_target_pdg

    @property
    def var_target_iscc(self):
        return self._dgen.var_target_iscc

    @property
    def data_loader(self):
        return self._dgen.data_loader

    @property
    def weights(self):
        return self._dgen.weights

    def __len__(self):
        return len(self._dgen)

    def __getitem__(self, index):
        return self._dgen.__getitem__(index)

