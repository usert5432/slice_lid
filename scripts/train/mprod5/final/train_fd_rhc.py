"""Train final MiniProd5 FD RHC SliceLID network"""

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
        'class_weights'   : 'equal',
        'dataset'         : 'mprod5/fd_rhc/dataset_slice_lid_fd_rhc.csv.xz',
        'data_mods'       : {
            'keep_pdg_iscc_list'    : None,
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
                'batchnorm'   : True,
                'layers_pre'  : [ 128, 128, 128 ],
                'lstm_units'  : 32,
                'layers_post' : [ 128, 128, 128 ],
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
        'target_pdg_iscc_list' : [ (12, 1), (14, 1), (16, 1), (0, 0) ],
        'test_size'       : 200000,
    # Args
        'outdir'          : 'mprod5/final/fd_rhc',
    }
)

setup_logging(
    logging.DEBUG, os.path.join(ROOT_OUTDIR, config['outdir'], "train.log")
)

parse_concurrency_cmdargs(config)
create_and_train_model(**config)

