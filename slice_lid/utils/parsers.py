"""
Functions to create standard `slice_lid` command line argument parsers.
"""

def parse_pdg_iscc_pair(pair):
    """Parse serialized (pdg,iscc) pair"""
    tokens = pair.split(',')

    if len(tokens) != 2:
        raise ValueError("Unknown pdg,iscc pair: '%s'" % (pair,))

    for i,t in enumerate(tokens):
        if t.lower() == 'none':
            tokens[i] = None
        else:
            tokens[i] = int(t)

    return tokens

def add_data_mods_parser(parser):
    """Create cmdargs parser of data modifications (balancer,filter)"""

    parser.add_argument(
        '--filter',
        default = None,
        help    = (
            'List of pairs of (pdg,iscc) that should be kept.'
            ' NOTE: -pdg and pdg treated separately'
        ),
        nargs   = '*',
        type    = parse_pdg_iscc_pair,
    )

    parser.add_argument(
        '--balance',
        default = None,
        help    = (
            'List of pairs of (pdg,iscc) that should be balanced',
            ' NOTE: -pdg and pdg treated as abs(pdg)'
        ),
        nargs   = '*',
        type    = parse_pdg_iscc_pair,
    )

def add_basic_eval_args(parser):
    """Create cmdargs parser of the standard evaluation options"""

    parser.add_argument(
        'outdir',
        help    = 'Directory with saved models',
        metavar = 'OUTDIR',
        type    = str,
    )

    parser.add_argument(
        '-e', '--ext',
        help    = 'Plot file extension',
        default = [ 'png' ],
        dest    = 'ext',
        nargs   = '+',
        type    = str,
    )

    parser.add_argument(
        '--class_weights',
        help    = 'Class weights to use for test dataset',
        default = 'none',
        dest    = 'class_weights',
        type    = str,
    )

    parser.add_argument(
        '-d', '--data',
        help    = 'Evaluate on a different dataset',
        default = None,
        dest    = 'data',
        type    = str,
    )

    parser.add_argument(
        '-t', '--test-size',
        help    = 'Take `test_size` subset from the dataset sample',
        default = 'same',
        dest    = 'test_size',
        type    = str,
    )

    add_data_mods_parser(parser)

