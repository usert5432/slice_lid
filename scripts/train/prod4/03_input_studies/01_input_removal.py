"""Train a family of SliceLID networks with removed inputs"""

import logging
import os

from speval import speval

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
        'dataset'         : 'prod4/fd_fhc/dataset_slice_lid_fd_fhc.csv.xz',
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
                'layers_pre'  : [ ],
                'lstm_units'  : 32,
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
        'target_pdg_iscc_list' : [ (12, 1), (14, 1), (16, 1), (0, 0) ],
        'test_size'       : 200000,
    # Args
        'outdir'          : 'prod4/03_input_studies/01_input_removal',
    }
)

VARS_SHOWER_LID_LL = [
    "png.shwlid.lid.eglll",
    "png.shwlid.lid.emulll",
    "png.shwlid.lid.epi0lll",
    "png.shwlid.lid.eplll",
    "png.shwlid.lid.enlll",
    "png.shwlid.lid.epilll",
    "png.shwlid.lid.egllt",
    "png.shwlid.lid.emullt",
    "png.shwlid.lid.epi0llt",
    "png.shwlid.lid.epllt",
    "png.shwlid.lid.enllt",
    "png.shwlid.lid.epillt",
]

VARS_SHOWER_LID_MISC = [
    "png.shwlid.calE",
    "png.shwlid.dir.x",
    "png.shwlid.dir.y",
    "png.shwlid.dir.z",
    "png.shwlid.nhit",
    "png.shwlid.nhitx",
    "png.shwlid.nhity",
    "png.shwlid.len",
    "png.shwlid.nplanex",
    "png.shwlid.nplaney",
    "png.shwlid.gap",
]

VARS_CVN = [
    "png.cvnpart.muonid",
    "png.cvnpart.electronid",
    "png.cvnpart.pionid",
    "png.cvnpart.protonid",
    "png.cvnpart.photonid",
]

search_space = [ ]

for cvn_vars in ([], VARS_CVN):
    for shwlid_ll_vars in ([], VARS_SHOWER_LID_LL):
        for shwlid_misc_vars in ([], VARS_SHOWER_LID_MISC):
            all_vars = cvn_vars + shwlid_ll_vars + shwlid_misc_vars
            if not all_vars:
                continue

            search_space.append({ 'vars_input_png3d' : all_vars })

parse_concurrency_cmdargs(config)

setup_logging(
    logging.DEBUG, os.path.join(ROOT_OUTDIR, config['outdir'], "train.log")
)

speval(
    lambda x : create_and_train_model(**config, extra_kwargs = x),
    search_space,
    os.path.join(ROOT_OUTDIR, config['outdir'], "trials.db"),
    timeout = 24 * 60 * 60
)

