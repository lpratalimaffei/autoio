"""
 tests pf reader
"""

import numpy
import mess_io
from _util import read_text_file


OUT_STR = read_text_file(['data', 'out'], 'messpf.dat')


def test__pf():
    """ test mess_io.reader.pfs.partition_function
    """

    ref_pf_dct = {
        100.0: (60.1983, 0.0302761, -0.000290427),
        200.0: (62.3581, 0.0163207, -6.51731e-05),
        298.2: (63.7335, 0.0123569, -2.32761e-05),
        300.0: (63.7557, 0.0123154, -2.28447e-05),
        400.0: (64.9053, 0.0109421, -6.74453e-06),
        500.0: (65.9811, 0.010705, 1.21039e-06),
        600.0: (67.0664, 0.0110813, 5.9895e-06),
        700.0: (68.2106, 0.011859, 9.42064e-06),
        800.0: (69.4485, 0.0129453, 1.22352e-05),
        900.0: (70.8085, 0.014296, 1.47438e-05),
        1000.0: (72.3157, 0.0158886, 1.70879e-05),
        1100.0: (73.9938, 0.0177105, 1.93376e-05),
        1200.0: (75.8652, 0.0197543, 2.15298e-05),
        1300.0: (77.9519, 0.0220153, 2.36856e-05),
        1400.0: (80.2754, 0.0244906, 2.58173e-05),
        1500.0: (82.8571, 0.0271782, 2.79324e-05),
        1600.0: (85.7181, 0.0300767, 3.00358e-05),
        1700.0: (88.8794, 0.033185, 3.21303e-05),
        1800.0: (92.3621, 0.0365025, 3.4218e-05),
        1900.0: (96.1869, 0.0400285, 3.63002e-05),
        2000.0: (100.375, 0.0437624, 3.83778e-05),
        2100.0: (104.946, 0.0477039, 4.04514e-05),
        2200.0: (109.922, 0.0518526, 4.25216e-05),
        2300.0: (115.324, 0.0562081, 4.45887e-05),
        2400.0: (121.171, 0.0607702, 4.6653e-05),
        2500.0: (127.485, 0.0655386, 4.87146e-05),
        2600.0: (134.285, 0.0705131, 5.07739e-05),
        2700.0: (141.594, 0.0756933, 5.2831e-05),
        2800.0: (149.431, 0.0810792, 5.4886e-05),
        2900.0: (157.817, 0.0866705, 5.6939e-05),
        3000.0: (166.772, 0.0924669, 5.89903e-05)
    }

    pf_dct = mess_io.reader.pfs.partition_function(OUT_STR)

    assert len(pf_dct) == len(ref_pf_dct)
    for temp in sorted(list(ref_pf_dct.keys())):
        assert numpy.allclose(pf_dct[temp], ref_pf_dct[temp])


def test__thermo():
    """ test mess_io.reader.pfs.entropy
        test mess_io.reader.pfs.heat_capacity
    """

    ref_s_dct = {
        100.0: 125.642,
        200.0: 130.404,
        298.2: 133.973,
        300.0: 134.037,
        400.0: 137.677,
        500.0: 141.754,
        600.0: 146.486,
        700.0: 152.044,
        800.0: 158.587,
        900.0: 166.278,
        1000.0: 175.279,
        1100.0: 185.754,
        1200.0: 197.865,
        1300.0: 211.779,
        1400.0: 227.657,
        1500.0: 245.665,
        1600.0: 265.967,
        1700.0: 288.727,
        1800.0: 314.109,
        1900.0: 342.276,
        2000.0: 373.392,
        2100.0: 407.622,
        2200.0: 445.127,
        2300.0: 486.072,
        2400.0: 530.619,
        2500.0: 578.931,
        2600.0: 631.171,
        2700.0: 687.501,
        2800.0: 748.084,
        2900.0: 813.082,
        3000.0: 882.657
    }

    ref_cp_dct = {
        100.0: 6.26152,
        200.0: 7.79248,
        298.2: 10.5319,
        300.0: 10.5982,
        400.0: 15.2508,
        500.0: 21.8742,
        600.0: 30.7096,
        700.0: 42.1657,
        800.0: 56.7204,
        900.0: 74.8681,
        1000.0: 97.1045,
        1100.0: 123.924,
        1200.0: 155.822,
        1300.0: 193.291,
        1400.0: 236.825,
        1500.0: 286.916,
        1600.0: 344.056,
        1700.0: 408.737,
        1800.0: 481.447,
        1900.0: 562.677,
        2000.0: 652.913,
        2100.0: 752.643,
        2200.0: 862.354,
        2300.0: 982.530,
        2400.0: 1113.66,
        2500.0: 1256.22,
        2600.0: 1410.71,
        2700.0: 1577.6,
        2800.0: 1757.37,
        2900.0: 1950.52,
        3000.0: 2157.52,
    }

    s_dct = mess_io.reader.pfs.entropy(OUT_STR)
    cp_dct = mess_io.reader.pfs.heat_capacity(OUT_STR)

    assert len(s_dct) == len(cp_dct)
    assert len(s_dct) == len(ref_s_dct)
    assert len(cp_dct) == len(ref_cp_dct)
    for temp in sorted(list(s_dct.keys())):
        assert numpy.allclose(s_dct[temp], ref_s_dct[temp])
        assert numpy.allclose(cp_dct[temp], ref_cp_dct[temp])


if __name__ == '__main__':
    test__pf()
    test__thermo()
