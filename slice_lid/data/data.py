"""
A collection of routines to simplify data handling.
"""

import logging
import os

from lstm_ee.data.data        import guess_data_loader, train_test_split
from lstm_ee.data.data_loader import DataShuffle

from .data_loader    import BalancedSampler, DataFilter
from .data_generator import (
    DataCache, DataClassWeights, DataDiskCache, DataGenerator, DataNANMask,
    MultiprocessedCache, MultithreadedCache
)

LOGGER = logging.getLogger('slice_lid.data')

def construct_data_loader(
    fname, seed, test_size, data_mods, var_pdg, var_iscc
):
    """Load dataset, transform/shuffle it and split into train/test parts.

    Parameters
    ----------
    path : str
        File path from which dataset will be loaded.
    seed : int or None
        Seed that will be used to initialize PRGs.
    test_size : int or float or None
        Fraction of the dataset that will go to the test sample.
        C.f. `train_test_split` for the detailed description.
    data_mods : dict
        Dictionary that specified data transformations to be applied to
        `data_loader`.
        C.f. `slice_lid.args.Config`.
    var_pdg : str
        Name of the variable in `data_loader` that defined particle PDG.
    var_iscc : str
        Name of the variable in `data_loader` that indicates whether event
        is Charged Current Event.

    Returns
    -------
    [ IDataLoader, IDataLoader ]
        A list of train and test DataLoaders.

    See Also
    --------
    add_data_modifiers
    train_test_split
    slice_lid.args.Config
    DataShuffle
    """

    data_loader = guess_data_loader(fname)
    data_loader = add_data_modifiers(
        data_loader, data_mods, seed, var_pdg, var_iscc
    )
    data_loader = DataShuffle(data_loader, seed)

    return train_test_split(data_loader, test_size)

def add_data_modifiers(data_loader, data_mods, seed, var_pdg, var_iscc):
    """Apply transformations to the `data_loader`.

    Parameters
    ----------
    data_loader : IDataLoader
        DataLoader over which transformations will be applied.
    data_mods : dict
        Dictionary that specified data transformations to be applied to
        `data_loader`.
        C.f. `slice_lid.args.Config`.
    seed : int or None
        Seed that will be used to initialize PRG.
    var_pdg : str
        Name of the variable in `data_loader` that defined particle PDG.
    var_iscc : str
        Name of the variable in `data_loader` that indicates whether event
        is Charged Current Event.

    Returns
    -------
    IDataLoader
        Decorated `data_loader` with transformations defined by `data_mods`
        applied on top of it.

    See Also
    --------
    DataFilter
    BalancedSampler
    slice_lid.args.Config
    """

    if data_mods is None:
        return data_loader

    keep_list = data_mods.get('keep_pdg_iscc_list', None)
    if keep_list is not None:
        LOGGER.info("Adding data filter with keep list: %s", keep_list)
        data_loader = DataFilter(data_loader, var_pdg, var_iscc, keep_list)

    balance_list = data_mods.get('balance_pdg_iscc_list', None)
    if balance_list is not None:
        LOGGER.info("Adding balanced sampler with targets: %s", balance_list)
        data_loader = BalancedSampler(
            data_loader, var_pdg, var_iscc, balance_list, seed
        )

    return data_loader

def add_cache_decorators(dgen_list, cache, concurrency, workers):
    """Add cache decorators to the DataGenerators from `dgen_list` list.

    Parameters
    ----------
    dgen_list : list of IDataGenerator
        A list of DataGenerators to be decorated.
    cache : bool or None
        If True then the DataGenerators from `dgen_list` will be cached.
        Otherwise, this function will return unmodified `dgen_list`.
    concurrency : { 'process', 'thread', None }
        Specifies Whether to precompute cache. If None, then cache will not be
        precomputed. Otherwise, it will be precomputed by parallelizing data
        generation in multiple threads/processes.
    workers : int or None
        Number of parallel threads/processes to use for precomputing cache.
        Has no effect if `concurrency` is None.

    Returns
    -------
    list of IDataGenerator
        DataGenerators from `dgen_list` decorated by cache decorators.

    See Also
    --------
    DataCache
    MultiprocessedCache
    MultithreadedCache
    """

    if (cache is None) or (not cache):
        return dgen_list

    if (workers is not None) and (workers > 0):
        if concurrency == 'process':
            LOGGER.info(
                "Using multiprocess data generator cache with %d workers",
                workers
            )
            return [
                MultiprocessedCache(x, workers) for x in dgen_list
            ]
        elif concurrency == 'thread':
            LOGGER.info(
                "Using multithreaded data generator cache with %d workers",
                workers
            )
            return [
                MultithreadedCache(x, workers) for x in dgen_list
            ]
        else:
            raise RuntimeError(
                "Unknown concurrency type: %s" % concurrency
            )
    else:
        LOGGER.info("Using data generator cache")
        return [ DataCache(x) for x in dgen_list ]

def add_disk_cache_decorators(dgen_list, disk_cache, **kwargs):
    """Add disk cache decorators to the DataGenerators from `dgen_list` list.

    Parameters
    ----------
    dgen_list : list of IDataGenerator
        A list of DataGenerators to be decorated.
    use_disk_cache : bool
        If True then disk cache decorators will be used. Otherwise, this
        function will return `dgen_list` unmodified.
    **kwargs : dict
        Dictionary that uniquely specifies given disk cache.
        C.f. DataDiskCache constructor.

    Returns
    -------
    list of IDataGenerator
        DataGenerators from `dgen_list` decorated by `DataDiskCache` decorators

    See Also
    --------
    DataDiskCache
    """

    if (not disk_cache) or (len(dgen_list) != 2):
        return dgen_list

    LOGGER.info("Using disk based data generator cache")
    return [
        DataDiskCache(dgen = dgen, part = idx, **kwargs) \
            for idx,dgen in enumerate(dgen_list)
    ]

def create_basic_data_generators(
    datadir              = None,
    dataset              = None,
    data_mods            = None,
    batch_size           = 1024,
    max_prongs           = None,
    seed                 = None,
    test_size            = 0.2,
    target_pdg_iscc_list = None,
    vars_input_slice     = None,
    vars_input_png3d     = None,
    var_target_pdg       = None,
    var_target_iscc      = None,
    disk_cache           = True,
):
    """
    Load dataset, shuffle, and create train/test DataGenerators.

    Parameters
    ----------
    datadir : str
        Root directory where datasets are located.
    dataset : str
        Path relative `datadir` to load dataset from.
    data_mods : dict
        Dictionary that specified data transformations to be applied to
        the dataset. C.f. `slice_lid.args.Config`.
    batch_size : int
        Size of the batches to be generated.
    max_prongs : int or None, optional
        If `max_prongs` is not None, then the number of 3D prongs will
        be truncated by `max_prongs`. Default: None.
    seed : int or None
        Seed that will be used initialize PRGs.
    test_size : int or float or None
        Amount of samples from `data_loader` that will go to the test sample.
        C.f. `train_test_split`.
    target_pdg_iscc_list : list of (int, bool)
        This parameters specifies target classes of the `slice_lid` network.
        The specification has the form
            [ (pdg_1, iscc_1), (pdg_2, iscc_2), ... (pdg_N, iscc_N) ]
        where pdg is an integer PDG of a particle and iscc is a boolean
        C.f. `slice_lid.args.Config`.
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

    Returns
    -------
    [ DataGenerator, DataGenerator ]
        Train and test DataGenerators.

    See Also
    --------
    slice_lid.args.Config
    construct_data_loader
    DataGenerator
    add_disk_cache_decorators
    """

    LOGGER.info("Loading %s dataset from %s.", dataset, datadir)
    path = os.path.join(datadir, dataset)
    data_loader_list = construct_data_loader(
        path, seed, test_size, data_mods, var_target_pdg, var_target_iscc
    )

    LOGGER.info(
          "Creating data generators with:\n"
        + "    batch size   : %d\n" % (batch_size)
        + "    max prongs   : %s\n" % (max_prongs)
        + "    seed         : %s\n" % (seed)
        + "    test size    : %s\n" % (test_size)
    )

    dgen_list = [
        DataGenerator(
            x, batch_size, max_prongs, target_pdg_iscc_list,
            vars_input_slice, vars_input_png3d,
            var_target_pdg, var_target_iscc
        )
        for x in data_loader_list
    ]

    return add_disk_cache_decorators(
        dgen_list, disk_cache,
        datadir              = datadir,
        dataset              = dataset,
        batch_size           = batch_size,
        max_prongs           = max_prongs,
        seed                 = seed,
        test_size            = test_size,
        target_pdg_iscc_list = target_pdg_iscc_list,
        vars_input_slice     = vars_input_slice,
        vars_input_png3d     = vars_input_png3d,
        var_target_pdg       = var_target_pdg,
        var_target_iscc      = var_target_iscc,
    )

def create_data_generators(
    datadir              = None,
    dataset              = None,
    data_mods            = None,
    batch_size           = 1024,
    class_weights        = None,
    max_prongs           = None,
    seed                 = None,
    test_size            = 0.2,
    target_pdg_iscc_list = None,
    vars_input_slice     = None,
    vars_input_png3d     = None,
    var_target_pdg       = None,
    var_target_iscc      = None,
    cache                = True,
    disk_cache           = True,
    concurrency          = None,
    workers              = 1,
):
    """
    Construct train/test DataGenerators from a dataset.

    Parameters
    ----------
    datadir : str
        Root directory where datasets are located.
    dataset : str
        Path relative `datadir` to load dataset from.
    data_mods : dict
        Dictionary that specified data transformations to be applied to
        the dataset. C.f. `slice_lid.args.Config`.
    batch_size : int
        Size of the batches to be generated.
    class_weights : { 'equal',  None }, optional
        Name of the class weights to use.
        C.f. `slice_lid.args.Config`.
    max_prongs : int or None, optional
        If `max_prongs` is not None, then the number of 3D prongs will
        be truncated by `max_prongs`. Default: None.
    seed : int or None
        Seed that will be used initialize PRGs.
    test_size : int or float or None
        Amount of samples from `data_loader` that will go to the test sample.
        C.f. `train_test_split`.
    target_pdg_iscc_list : list of (int, bool)
        This parameters specifies target classes of the `slice_lid` network.
        The specification has the form
            [ (pdg_1, iscc_1), (pdg_2, iscc_2), ... (pdg_N, iscc_N) ]
        C.f. `slice_lid.args.Config`.
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
    cache : bool or None
        Specifies whether to cache batches in RAM. C.f. `add_cache_decorators`.
    disk_cache : bool or None
        Specifies whether to cache batches on disk.
        C.f. `add_disk_cache_decorators`.
    concurrency : { None, 'thread', 'process' }
        If not None then batches will be precomputed in parallel.
        C.f. `add_cache_decorators`.
    workers : int or None
        Number of parallel threads/processes to use for precomputing batches.
        C.f. `add_cache_decorators`.

    Returns
    -------
    [ DataGenerator, DataGenerator ]
        Train and test DataGenerators.

    See Also
    --------
    slice_lid.args.Config
    create_basic_data_generators
    add_cache_decorators
    """

    dgen_list = create_basic_data_generators(
        datadir, dataset, data_mods, batch_size, max_prongs, seed, test_size,
        target_pdg_iscc_list, vars_input_slice, vars_input_png3d,
        var_target_pdg, var_target_iscc, disk_cache
    )

    if class_weights is not None:
        dgen_list = [
            DataClassWeights(dgen, class_weights) for dgen in dgen_list
        ]

    dgen_list = add_cache_decorators(dgen_list, cache, concurrency, workers)
    dgen_list = [ DataNANMask(x) for x in dgen_list ]

    # pylint: disable = import-outside-toplevel
    from .data_generator.keras_sequence import KerasSequence
    dgen_list = [ KerasSequence(x) for x in dgen_list ]

    return dgen_list

def load_data(args):
    """
    Wrapper around `create_data_generators` that unpacks arguments from `args`.
    """

    return create_data_generators(
        datadir              = args.root_datadir,
        dataset              = args.dataset,
        data_mods            = args.data_mods,
        batch_size           = args.batch_size,
        class_weights        = args.class_weights,
        max_prongs           = args.max_prongs,
        seed                 = args.seed,
        test_size            = args.test_size,
        target_pdg_iscc_list = args.target_pdg_iscc_list,
        vars_input_slice     = args.vars_input_slice,
        vars_input_png3d     = args.vars_input_png3d,
        var_target_pdg       = args.var_target_pdg,
        var_target_iscc      = args.var_target_iscc,
        cache                = args.cache,
        disk_cache           = args.disk_cache,
        concurrency          = args.concurrency,
        workers              = args.workers,
    )

