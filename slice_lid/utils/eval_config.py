"""
Definition of `EvalConfig` that holds parameters of evaluation.
"""

from lstm_ee.utils.eval_config import build_eval_subdir, modify_args_value

def modify_dict_value(d, key, eval_value):
    """Modify single value in a dict `d` with `key` to `eval_value`.

    Attention is paid to a special value of `eval_value` == 'none', in which
    case d[key] is set to None.
    """
    if eval_value is None:
        return

    if eval_value == 'none':
        d[key] = None
    else:
        d[key] = eval_value

class EvalConfig:
    """Configuration of a `slice_lid` network evaluation.

    Parameters of the `EvalConfig` will be used to modify the corresponding
    parameters of the `Args` of the trained network. The rules of such
    modification are as follows:
        - If (parameter == "same") or (parameter is None) then the value of
          `Args` will not be modified.
        - If parameter == "none" then the value of `Args` will be set to None.
        - otherwise `Args` parameter will be either directly set to parameter
          from `EvalConfig` or will be loaded from a file specified by a
          parameter of `EvalConfig`.

    Parameters
    ----------
    data : str or None,
        Name of the evaluation dataset.
    balance_pdg_iscc_list : list of (int, bool) or None
        Targets to be balanced.
        C.f. `Config.data_mods`.
    class_weights : str or None,
        Name of the class weights to use.
        C.f. `Config.class_weights`.
    keep_pdg_iscc_list : list of (int, bool) or None
        If defined, then the dataset will be filtered and only targets
        specified by `keep_pdg_iscc_list` will be kept.
        C.f. `Config.data_mods`.
    test_size : int or float or None
        Size of the dataset that will be used for evaluation.
        C.f. `Config.test_size`.

    See Also
    --------
    slice_lid.args.Config
    """

    @staticmethod
    def _recognize_same(value):
        if isinstance(value, str) and (value == 'same'):
            return None

        return value

    @staticmethod
    def from_cmdargs(cmdargs):
        """Construct `EvalConfig` from parameters from `argparse.Namespace`"""
        return EvalConfig(
            cmdargs.data,
            cmdargs.balance,
            cmdargs.class_weights,
            cmdargs.filter,
            cmdargs.test_size,
        )

    def __init__(
        self, data, balance_pdg_iscc_list, class_weights,
        keep_pdg_iscc_list, test_size
    ):
        self.data          = EvalConfig._recognize_same(data)
        self.balance_list  = EvalConfig._recognize_same(balance_pdg_iscc_list)
        self.class_weights = EvalConfig._recognize_same(class_weights)
        self.keep_list     = EvalConfig._recognize_same(keep_pdg_iscc_list)
        self.test_size     = EvalConfig._recognize_same(test_size)

    def get_eval_subdir(self):
        """Create eval subdir that is unique for this evaluation config."""
        return build_eval_subdir([
            ('data',    self.data),
            ('balance', self.balance_list),
            ('cweight', self.class_weights),
            ('filter',  self.keep_list),
            ('tsize',   self.test_size),
        ])

    def modify_eval_args(self, args):
        """Modify parameters of `args` using values from `self`"""
        modify_args_value(args.config, 'dataset',       self.data)
        modify_args_value(args.config, 'class_weights', self.class_weights)
        modify_args_value(args.config, 'test_size',     self.test_size, float)

        if args.config.data_mods is None:
            args.config.data_mods = {}

        modify_dict_value(
            args.config.data_mods, 'keep_pdg_iscc_list', self.keep_list
        )
        modify_dict_value(
            args.config.data_mods, 'balance_pdg_iscc_list', self.balance_list
        )

