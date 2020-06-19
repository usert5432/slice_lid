"""
Definition of a `Config` class that parametrizes training.
"""

import json

class Config:
    """Training configuration.

    `Config` is a structure that holds parameters required to reproduce the
    training.

    Parameters
    ----------
    batch_size : int
        Training batch size.
    class_weights : { 'equal',  None }, optional
        Name of the class weights to use.
        If 'equal' then the weights that make targets equally represented
        will be used. If None, then no class weights will be used.
        Default: None
    dataset : str
        Dataset path inside "${SLICE_LID_DATADIR}".
    data_mods : dict or None, optional
        Specification of the dataset transformations of the form
        {
            'keep_pdg_iscc_list'    : KEEP_PDG_ISCC_LIST,
            'balance_pdg_iscc_list' : BALANCE_PDG_ISCC_LIST,
        }.
        If 'keep_pdg_iscc_list' is present in `data_mods` and its value
        KEEP_PDG_ISCC_LIST is not None, then the dataset will be filtered
        according to the following rules:
            - first, KEEP_PDG_ISCC_LIST is expected to be a list of pairs
              [ (pdg_0, iscc_0), (pdg_1, iscc_1), ... (pdg_n, iscc_N) ].
            - Only samples in the dataset that have (pdg, iscc) values matching
              values in the KEEP_PDG_ISCC_LIST will be kept. Other samples
              will be discarded.
            - Matching of sample values (pdg, iscc) to the pairs in
              KEEP_PDG_ISCC_LIST performed according to the rules defined
              by `slice_lis.data.data_loader.DataFilter`
        If 'balance_pdg_iscc_list' is present in `data_mods` and its value
        BALANCE_PDG_ISCC_LIST is not None, then the subsample of the dataset
        will be selected that has equal number of targets (as specified by
        the BALANCE_PDG_ISCC_LIST). The subsample will be selected according
        to the following rules:
            - The BALANCE_PDG_ISCC_LIST is expected to be a list of pairs
              [ (pdg_0, iscc_0), (pdg_1, iscc_1), ... (pdg_n, iscc_N) ].
            - The BALANCE_PDG_ISCC_LIST defines targets similar to
              `target_pdg_iscc_list`, except that it also allows for wildcard
              targets.
            - The detailed rules on selecting the subsample are defined at
              `slice_lid.data.data_loader.BalancedSampler`.
        If `data_mods` is None, no data transformations will be applied.
    early_stop : dict or None, optional
        Early stopping configuration.
        C.f. `lstm_ee.train.setup.get_early_stop` for available configurations.
        If None, no early stopping will be used. Default: None.
    epochs : int
        Number of epochs training will be run.
    max_prongs : int or None, optional
        Limit number of 3D prongs to `max_prongs`. In other words, if the
        number of 3D prongs is greater than `max_prongs` the remaining prongs
        will be discarded. If None no prong limit will be applied.
        Default: None.
    model : dict
        Network configuration. The `model` dict is expected to have the
        following form: { 'name' : NAME, 'kwargs' : KWARGS_DICT }.
        Where NAME specifies the name of the network and KWARGS_DICT is a
        dict of network parameters.
        C.f. `slice_lid.train.setup.select_model` for available configurations.
    optimizer : dict
        Optimizer configuration to use for training.
        C.f. `lstm_ee.train.setup.get_optimizer` for the available options.
    regularizer : dict or None, optional
        Regularization configuration to be used during the training.
        C.f. `lstm_ee.train.setup.get_regularizer` for the available options.
        If None, no regularization will be used. Default: None.
    schedule : dict or None, optional
        Learning rate decay schedule configuration.
        C.f. `lstm_ee.train.setup.get_schedule` for the available options.
        If None, no schedule will be used. Default: None.
    seed : int
        Seed to be used for the pseudo-random generator initialization.
        This seed affects:
          - data shuffling
          - data train/test split
          - training itself
    steps_per_epoch : int or None, optional
        Number of batches to use per training epoch. If None then all available
        batches will be used in a single epoch. Default: None.
    target_pdg_iscc_list : list of (int, bool)
        This parameters specifies target classes of the `slice_lid` network.
        The specification has the form
            [ (pdg_1, iscc_1), (pdg_2, iscc_2), ... (pdg_N, iscc_N) ]
        where pdg is an integer PDG of a particle and iscc is a boolean
        flag indicating whether event is a Charged Current event.
        The `slice_lid` network with the above `target_pdg_iscc_list` is going
        to have have N+1 targets.
        If given data sample has pdg and iscc matching one of the pairs in
        `target_pdg_iscc_list` (say at Mth position), then the target array
        for this sample will be one-hot encoding of M, that is:
            [ 0, ..., 0, 1 (at Mth index), 0, ..., 0 ]
        If given data sample has pdg and iscc that does not match any of
        the pairs in the `target_pdg_iscc_list`, then the target array
        for this sample will be one-hot encoding of 0, that is:
            [ 1, 0, ..., 0 ]
    test_size : int or float
        Amount of the `dataset` to be used for network validation
        (aka validation set or dev set).
        If `test_size` is int, then `test_size` entries will be uniformly
        randomly taken from the `dataset` as a validation sample.
        If `test_size` is float and `test_size` < 1, then a fraction
        `test_size` of the `dataset` will be held as validation sample
        (also sampled uniformly at random).
    vars_input_slice : list of str or None, optional
        Names of slice level input variables. If None then no slice level
        inputs will be used. Default: None.
    vars_input_png3d : list of str or None, optional
        Names of 3D prong level input variables. If None then no 3D prong level
        inputs will be used. Default: None.
    var_target_pdg : str
        Name of the variable that has true PDG number of the event.
    var_target_iscc : str
        Name of the variable that specifies whether given event was a Charged
        Current event.
    """

    __slots__ = (
        'batch_size',
        'class_weights',
        'dataset',
        'data_mods',
        'early_stop',
        'epochs',
        'max_prongs',
        'model',
        'optimizer',
        'regularizer',
        'schedule',
        'seed',
        'steps_per_epoch',
        'target_pdg_iscc_list',
        'test_size',
        'vars_input_slice',
        'vars_input_png3d',
        'var_target_pdg',
        'var_target_iscc',
    )

    def __init__(self, **kwargs):

        for k in self.__slots__:
            setattr(self, k, None)

        for k,v in kwargs.items():
            if k in self.__slots__:
                setattr(self, k, v)

    def save(self, savedir):
        """Save configuration to `savedir`/config.json"""
        kwargs = { x : getattr(self, x) for x in self.__slots__ }

        with open("%s/config.json" % (savedir), 'wt') as f:
            json.dump(kwargs, f, sort_keys = True, indent = 4)

    @staticmethod
    def load(savedir):
        """Load configuration from `savedir`/config.json"""
        with open("%s/config.json" % (savedir), 'rt') as f:
            kwargs = json.load(f)

        return Config(**kwargs)

    def __str__(self):
        kwargs = { x : getattr(self, x) for x in self.__slots__ }
        return json.dumps(kwargs, sort_keys = True)

    def pprint(self):
        """Pretty print configuration"""
        kwargs = { x : getattr(self, x) for x in self.__slots__ }
        return json.dumps(kwargs, sort_keys = True, indent = 4)


