""" reaction read write testing
"""

import numpy as np
from chemkin_io.writer.mechanism import reactions_block as writer
from chemkin_io.parser.reaction import param_dct as parser


ARRHENIUS_RXN_PARAM_DCT = {
    (('H', 'O2'), ('OH', 'O'), (None,)): (
        ([1E+15, 0.00, 25000], None, None, None, None, None),)
}

LINDEMANN_RXN_PARAM_DCT = {
    (('H', 'O2'), ('OH', 'O'), ('(+M)',)): (
        ([1E+15, 0.00, 25000],
         [1E+15, 0.00, 25000],
         None,
         None,
         None,
         {'AR': 1.4, 'N2': 1.7}),
    )
}

TROE_RXN_PARAM_DCT = {
    (('H', 'O2'), ('OH', 'O'), ('(+AR)',)): (
        ([1E+15, 0.00, 25000],
         [1E+15, 0.00, 25000],
         [0.95, 1E-30, 8000],
         None,
         None,
         None),
    )
}

CHEBYSHEV_RXN_PARAM_DCT = {
    (('H', 'O2'), ('OH', 'O'), ('(+N2)',)): (
        ([1E+15, 0.00, 25000],
         None,
         None,
         {'t_limits': [500.0, 2000.0],
          'p_limits': [0.03, 100.0],
          'alpha_elm': np.array([
              [-1.620e+01, -1.183e-01, -5.423e-02, -1.476e-02],
              [2.578e+00, 1.614e-01, 7.359e-02, 1.880e-02],
              [1.068e-01, -7.235e-02, -2.733e-02, -3.778e-03],
              [3.955e-02, 1.207e-02, 3.402e-04, -2.695e-03],
              [8.557e-03, 4.345e-03, 3.670e-03, 1.608e-03],
              [8.599e-04, -1.758e-03, -7.502e-04, 7.396e-07]]),
          'a_units': 'moles'},
         None,
         None),
    )
}

PLOG_RXN_PARAM_DCT = {
    (('H', 'O2'), ('OH', 'O'), (None,)): (
        ([1E+15, 0.00, 25000],
         None,
         None,
         None,
         {0.1: [1E+15, 0.00, 25000],
          1.0: [1E+15, 0.00, 25000],
          10.0: [1E+15, 0.00, 25000],
          100.0: [1E+15, 0.00, 25000]},
         None),
    )
}

DUPLICATE_ARRHENIUS_RXN_PARAM_DCT = {
    (('H', 'O2'), ('HO2'), ('+M',)): (
        ([1E+15, 0.00, 25000], None, None, None, None, None),
        ([1E+15, 0.00, 25000], None, None, None, None, None),
    )
}

DUPLICATE_PLOG_RXN_PARAM_DCT = {
    (('H', 'O2'), ('OH', 'O'), (None,)): (
        ([1E+15, 0.00, 25000],
         None,
         None,
         None,
         {0.1: [1E+15, 0.00, 25000],
          1.0: [1E+16, 0.00, 25000],
          10.0: [1E+17, 0.00, 25000],
          100.0: [1E+18, 0.00, 25000]},
         None),
        ([1E+15, 0.00, 25000],
         None,
         None,
         None,
         {0.1: [1E+15, 0.00, 25000],
          1.0: [1E+16, 0.00, 25000],
          100.0: [1E+18, 0.00, 25000]},
         None),
    )
}


def test__arrhenius():
    """ Tests the Arrhenius reader and writer
    """
    first_str, second_str = _get_strs(ARRHENIUS_RXN_PARAM_DCT)
    assert first_str == second_str


def test__lindemann():
    """ Tests the Lindemann reader and writer
    """
    first_str, second_str = _get_strs(LINDEMANN_RXN_PARAM_DCT)
    assert first_str == second_str


def test__troe():
    """ Tests the Troe reader and writer
    """
    first_str, second_str = _get_strs(TROE_RXN_PARAM_DCT)
    assert first_str == second_str


def test__plog():
    """ Tests the PLOG reader and writer
    """
    first_str, second_str = _get_strs(PLOG_RXN_PARAM_DCT)
    assert first_str == second_str


def test__chebyshev():
    """ Tests the Chebyshev reader and writer
    """
    first_str, second_str = _get_strs(CHEBYSHEV_RXN_PARAM_DCT)
    assert first_str == second_str


def test__dup_arrhenius():
    """ Tests the Arrhenius reader and writer for a duplicate reaction
    """
    first_str, second_str = _get_strs(DUPLICATE_ARRHENIUS_RXN_PARAM_DCT)
    assert first_str == second_str


def test__dup_plog():
    """ Tests the PLOG reader and writer for a duplicate reaction
    """
    first_str, second_str = _get_strs(DUPLICATE_PLOG_RXN_PARAM_DCT)
    assert first_str == second_str


def _get_strs(rxn_param_dct):
    """ Gets two strs after passing a rxn_param_dct through
        the writer and reader
    """
    first_str = writer(rxn_param_dct)
    new_rxn_param_dct = parser(first_str, 'cal/mole', 'moles')
    second_str = writer(new_rxn_param_dct)

    return first_str, second_str

test__dup_plog()
#  test__dup_arrhenius()
test__plog()
#test__arrhenius()
