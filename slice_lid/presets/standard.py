"""
Standard `slice_lid` training/evaluation presets.
"""

STANDARD_PRESET = {
    'var_target_pdg'   : 'mc.pdg',
    'var_target_iscc'  : 'mc.isCC',
    'vars_input_png3d' : [
        "png.cvnpart.muonid",
        "png.cvnpart.electronid",
        "png.cvnpart.pionid",
        "png.cvnpart.protonid",
        "png.cvnpart.photonid",
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
    ],
    'vars_input_slice' : [
        "calE"
    ],
}


def add_train_standard_presets(presets):
    """Add standard training presets to the `presets` dict"""
    presets['standard'] = STANDARD_PRESET

def add_eval_standard_presets(_presets):
    """Add standard evaluation presets to the `presets` dict"""
    # pylint: disable=unnecessary-pass
    pass

