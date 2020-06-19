"""
Definition of the `Args` object that holds runtime training configuration.
"""
import copy
import json
import os

from lstm_ee.args.funcs import calc_savedir, update_kwargs
from slice_lid.consts   import ROOT_DATADIR, ROOT_OUTDIR, DEF_SEED

from .config import Config

class Args:
    """Runtime training configuration.

    `Args` contains an instance of `Config` that defines the training, plus a
    number of options that are not necessary for the reproduction of training.

    Parameters
    ----------
    outdir : str
        Parent directory under `root_outdir` where model directory will be
        created.
    cache : bool, optional
        If True data batches will be cached in RAM. Default: False.
        If `cache` is False and `concurrency` is not None, then the internal
        `keras` concurrent data generation will be used.
        Otherwise, data cache will be filled in parallel and keras will be
        run without concurrent data generation.
    disk_cache : bool, optional
        If True data batches will be cached in on a disk. Default: False.
        Caches are stored under "`root_outdir`/.cache" and should be
        cleaned manually.
    concurrency : { 'process', 'thread', None}, optional
        Type of the parallel data batch generation to use.
        If `concurrency` is "process" then will spawn several parallel
        processes for the data batch generation (may eat all your RAM).
        If "thread" then will spawn several parallel threads, mostly
        ineffective due to GIL.
        The number of parallel threads or processes is controlled by the
        `workers` parameter.
        If None then will not use parallelized data batch generation.
        Default: None.
    workers : int or None, optional
        Number of parallel workers to spawn for the purpose of data batch
        generation. If None then no parallelization will be used.
    **kwargs : dict
        Parameters to be passed to the `Config` constructor.
    extra_kwargs : dict or None, optional
        Specifies extra arguments that will be used to modify `kwargs` above.
        This `extra_kwargs` will be saved in a separate file and will be
        used to determine model `savedir`.

    Attributes
    ----------
    config : Config
        Training configuration.
    savedir : str
        Directory under `root_outdir` where trained model and its config
        will be saved.  It is calculated based on the `outdir` and
        `extra_kwargs` parameters following the pattern:
        `savedir` = `outdir`/model_`extra_kwargs`_hash(hash of `config`).
    root_data : str
        Parent directory where all data is saved.
        Unless set explicitly it is equal to "${SLICE_LID_DATADIR}".
    root_outdir : str
        Parent directory where all trained models are saved.
        Unless set explicitly it is equal to "${SLICE_LID_OUTDIR}".

    See Also
    --------
    lstm_ee.args.Args : Similar structure for the `lstm_ee`.
    """

    # pylint: disable=access-member-before-definition

    __slots__ = (
        'config',
        'save_best',
        'savedir',
        'outdir',

        'root_outdir',
        'root_datadir',

        'cache',
        'disk_cache',
        'concurrency',
        'workers',

        'extra_kwargs',
    )

    def __init__(
        self, loaded = False, extra_kwargs = None, **kwargs
    ):
        for k in self.__slots__:
            setattr(self, k, None)

        self.extra_kwargs = extra_kwargs

        kwargs = copy.deepcopy(kwargs)
        update_kwargs(kwargs, extra_kwargs)

        self.config = Config(**kwargs)

        for k,v in kwargs.items():
            if k in self.__slots__:
                setattr(self, k, v)
            else:
                if not k in self.config.__slots__:
                    raise ValueError(
                        "Unknown Parameter '%s = %s'" % (k, v)
                    )

        self._init_default_values()

        if not loaded:
            self._init_savedir()

    @staticmethod
    def load(savedir):
        """Load `Args` from the directory `savedir`"""
        # pylint: disable=attribute-defined-outside-init

        config = Config.load(savedir)
        result = Args(loaded = True)

        result.config  = config
        result.savedir = savedir

        result.outdir = os.path.normpath(
            os.path.join(result.savedir, os.path.pardir)
        )

        try:
            with open("%s/extra.json" % (savedir), 'rt') as f:
                result.extra_kwargs = json.load(f)
        except IOError:
            pass

        return result

    def _init_default_values(self):
        if self.root_datadir is None:
            self.root_datadir = ROOT_DATADIR

        if self.root_outdir is None:
            self.root_outdir = ROOT_OUTDIR

        if self.config.seed is None:
            self.config.seed = DEF_SEED

    def _init_savedir(self):

        self.savedir = calc_savedir(
            os.path.join(self.root_outdir, self.outdir),
            'model', self.config, self.extra_kwargs
        )

        self.config.save(self.savedir)

        with open("%s/extra.json" % (self.savedir), 'wt') as f:
            json.dump(self.extra_kwargs, f, sort_keys = True, indent = 4)

    def __getattr__(self, name):
        """Get attribute `name` from `Args.config`.

        This function is invoked when one has called `Args.name`, but the
        `Args` itself does not have `name` attribute. In such case it will
        return `Args.config.name`.
        """
        return getattr(self.config, name)

    def __getitem__(self, name):
        """Get `Args` or `Args.config` attribute specified by address `name`.

        See Also
        --------
        lstm_ee.args.Args.__getitem__
        """

        if isinstance(name, list):
            return [self[n] for n in name]

        address = name.split(':')

        result = getattr(self, address[0])

        for addr_part in address[1:]:
            result = result[addr_part]

        return result

