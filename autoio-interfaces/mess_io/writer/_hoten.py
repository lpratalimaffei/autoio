""" Write BF derived from hotenergies
"""

import pandas as pd
from ratefit.fit import arrhenius as arrfit
# chemkin_str = arrfit.pes(
#    ktp_dct, reaction, mess_path)
# ktp_dct, rxn_name(str), fakepath, **kwargs

DEFAULT_ARRFIT_DCT = {
    'dbltol': 10000.0,
    'dblcheck': 'max'
}


def rename_ktp_dct(ktp_dct, pedspecies, label_dct):
    """ rename ktp dictionary with rxn names
        ktp_dct.keys(): sp
        renamed_ktp_dct.keys(): rctname=>sp
        if sp is the origina produce, the reaction is reversible =
    """
    rename_ktp_dct = {}
    print(label_dct, pedspecies)
    # revers label dictionary
    label_dct_rev = dict(zip(label_dct.values(), label_dct.keys()))
    reacs = label_dct_rev[pedspecies[0]]
    prods = label_dct_rev[pedspecies[1]]
    prods_lst = prods.split('+')
    for pi in prods_lst:
        # if pi is in label_dct keys: is the product
        if pi in label_dct.keys():
            prod1 = pi
    if all(pi in label_dct.keys() for pi in prods_lst):
        print('*Warning: only prompt product should be specified in label dct')
    for sp in ktp_dct.keys():
        linker = (label_dct_rev[sp] == prod1)*'=' + \
            (label_dct_rev[sp] != prod1)*'=>'
        prodsnew = prods.replace(prod1, label_dct_rev[sp])
        newkey = linker.join([reacs, prodsnew])
        rename_ktp_dct[newkey] = ktp_dct[sp]

    return rename_ktp_dct


def fit_ktp_dct(ktp_dct, mess_path):
    """ Fit with plog a given ktp dictionary
        returns strings with fits
    """
    chemkin_str = ''
    for reaction in ktp_dct.keys():

        chemkin_str += arrfit.pes(
            ktp_dct[reaction], reaction, mess_path, **DEFAULT_ARRFIT_DCT)

    return chemkin_str
