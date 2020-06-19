"""Train simplest SliceLID network without optimizations."""

import logging
import os

from slice_lid.args    import join_dicts
from slice_lid.consts  import ROOT_OUTDIR
from slice_lid.presets import PRESETS_TRAIN
from slice_lid.train   import create_and_train_model
from lstm_ee.utils     import setup_logging, parse_concurrency_cmdargs

config = join_dicts(
    PRESETS_TRAIN['standard'],
    {
    # Config
        'batch_size'      : 1024,
        'class_weights'   : None,
        'dataset'         : 'prod4/fd_fhc/dataset_slice_lid_fd_fhc.csv.xz',
        'data_mods'       : {
            'keep_pdg_iscc_list' : [
                (12, None), (-12, None),
                (14, None), (-14, None),
                (16, None), (-16, None),
            ],
            'balance_pdg_iscc_list' : None,
        },
        'early_stop'   : {
            'name'   : 'standard',
            'kwargs' : {
                'monitor'   : 'val_loss',
                'min_delta' : 0,
                'patience'  : 40,
            },
        },
        'epochs'       : 200,
        'max_prongs'   : None,
        'model'        : {
            'name'   : 'standard',
            'kwargs' : {
                'batchnorm'   : False,
                'layers_pre'  : [ ],
                'lstm_units'  : 16,
                'layers_post' : [ ],
                'n_resblocks' : 0,
            },
        },
        'optimizer'      : {
            'name'   : 'RMSprop',
            'kwargs' : {
                'lr'        : 0.001,
                'clipnorm'  : 0.5,
                'clipvalue' : 0.5,
            },
        },
        'regularizer'    : {
            'name'   : 'l1',
            'kwargs' : { 'l' : 0.0001 },
        },
        'schedule'       : {
            'name'   : 'standard',
            'kwargs' : {
                'monitor'  : 'val_loss',
                'factor'   : 0.5,
                'patience' : 5,
                'cooldown' : 0
            },
        },
        'seed'            : 0,
        'steps_per_epoch' : 500,
        'target_pdg_iscc_list' : [ (12, 1) ],
        'test_size'       : 200000,
    # Args
        'outdir'          : 'prod4/01_initial_studies/01_original_nue',
    }
)

parse_concurrency_cmdargs(config)

setup_logging(
    logging.DEBUG, os.path.join(ROOT_OUTDIR, config['outdir'], "train.log")
)

create_and_train_model(**config)

