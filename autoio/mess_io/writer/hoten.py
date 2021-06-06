""" Write BF derived from hotenergies
"""

import pandas as pd
from ratefit.fit import arrhenius as arrfit
#chemkin_str = arrfit.pes(
#    ktp_dct, reaction, mess_path)
# ktp_dct, rxn_name(str), fakepath, **kwargs

DEFAULT_ARRFIT_DCT = {
    'dbltol': 100.0,
    'dblcheck': 'max'
}

def rename_ktp_dct(ktp_dct, rctname, radname):
    """ rename ktp dictionary with rxn names
        ktp_dct.keys(): sp
        renamed_ktp_dct.keys(): rctname=>sp
        if sp in radname, the reaction is reversible =
        TO REVISE WITH THE NAME OF THE SECOND REACTANT
    """
    rename_ktp_dct = {}

    for sp in ktp_dct.keys():
        linker = (sp in radname)*'=' + (sp not in radname)*'=>'
        newkey = linker.join([rctname, sp])
        rename_ktp_dct[newkey] = ktp_dct[sp]

    return rename_ktp_dct

def fit_ktpdct(ktp_dct, mess_path):
    """ Fit with plog a given ktp dictionary
        returns strings with fits
    """
    chemkin_str = ''
    for reaction in ktp_dct.keys():

        chemkin_str += arrfit.pes(
            ktp_dct[reaction], reaction, mess_path, **DEFAULT_ARRFIT_DCT)

    return chemkin_str