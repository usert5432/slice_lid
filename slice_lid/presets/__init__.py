"""A collection of training/evaluation presets"""

from .standard import add_train_standard_presets,add_eval_standard_presets

PRESETS_TRAIN = {}
PRESETS_EVAL  = {}

add_train_standard_presets(PRESETS_TRAIN)
add_eval_standard_presets(PRESETS_EVAL)

