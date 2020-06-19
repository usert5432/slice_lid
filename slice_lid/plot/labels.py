"""Functions to calculate human readable labels for slice_lid targets."""
PDG_NAME_MAP = {
    0  : ('Cosmics' , 'Cosmics'),
    12 : ('NuE'     , 'Electron Neutrino'),
    14 : ('NuMu'    , 'Muon Neutrino'),
    16 : ('NuTau'   , 'Tau Neutrino'),
}

def convert_targets_to_labels(targets_pdg_iscc_list):
    """Function to convert list of (pdg,iscc) pairs to a list of str names"""
    labels = [ 'Bkg' ]

    for (pdg, iscc) in targets_pdg_iscc_list:
        if pdg == 0:
            label = "%s" % (PDG_NAME_MAP[pdg][0])
        else:
            label = "%s:%s" % (PDG_NAME_MAP[pdg][0], "CC" if iscc else "NC")

        labels.append(label)

    return labels

