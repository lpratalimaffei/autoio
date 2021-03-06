"""
  Additional functions for formatting information for MESS strings
"""

import numpy
from ioformat import indent


# Format various pieces of data into strings for MESS input files
def zero_energy_format(zero_ene):
    """ Formats the zero point energy into a string that
        is appropriate for a MESS input file.

        :param zero_ene: zero point energy value
        :type zero_ene: float
        :return zero_ene_str: MESS-format string containing energy
        :rtype string
    """

    zero_ene_str = (
        'ZeroEnergy[kcal/mol]      {0:<8.2f}'.format(zero_ene)
    )

    return zero_ene_str


def elec_levels_format(elec_levels):
    """ Formats the list of electronic energy levels into a string that
        is appropriate for a MESS input file.

        :param elec_levels: levels, given as [[energy, degeneracy], ...]
        :type elec_levels: list(list(float))
        :return elec_levels_str: MESS-format string containing levels
        :rtype string
    """

    # Get the number of elec levles
    nlevels = len(elec_levels)

    # Build elec levels string
    elec_levels_str = ''
    for i, level in enumerate(elec_levels):
        elec_levels_str += '  '.join(map(str, level))
        if (i+1) != len(elec_levels):
            elec_levels_str += '\n'

    # Indent the lines
    elec_levels_str = indent(elec_levels_str, 4)

    return nlevels, elec_levels_str


def geometry_format(geo):
    """ Formats the geometry of a species into a string that
        is appropriate for a MESS input file.

        :param geo: geometry of a species
        :return natoms: number of atoms in the geometry
        :rtype int
        :return geo_str: MESS-format string containing geometry
        :rtype string
    """

    # Get the number of atoms
    natoms = len(geo)

    # Build geom string; converting the coordinates to angstrom
    geo_str = ''
    for (asymb, xyz) in geo:
        geo_str += '{:<4s}{:>14.5f}{:>14.5f}{:>14.5f}\n'.format(
            asymb, *tuple((val*0.529177 for val in xyz)))

    # Remove final newline character and indent the lines
    geo_str = indent(geo_str.rstrip(), 4)

    return natoms, geo_str


def freqs_format(freqs):
    """ Formats the vibrational frequencies of a species into a string that
        is appropriate for a MESS input file.

        :param freqs: vibrational frequencies of species
        :type freqs: list(float)
        :return nfreqs: number of frequences for the species
        :rtype int
        :return freq_str: MESS-format string containing frequencies
        :rtype string
    """

    # Get the number of freqs
    nfreqs = len(freqs)

    # Build freqs string
    freq_str = ''
    for i, freq in enumerate(freqs):
        if ((i+1) % 6) == 0 and (i+1) != len(freqs):
            freq_str += '{0:<8.0f}\n'.format(int(freq))
        else:
            freq_str += '{0:<8.0f}'.format(freq)

    # Indent the lines
    freq_str = indent(freq_str, 4)

    return nfreqs, freq_str


def intensities_format(intens):
    """ Formats the vibrational intenuencies of a species into a string that
        is appropriate for a MESS input file.

        :param intens: harmonic infrared intensities of species
        :type intens: list(float)
        :return nintens: number of harmonic infrared intensities for species
        :rtype int
        :return inten_str: MESS-format string containing infrared intensities
        :rtype string
    """

    # Get the number of intens
    nintens = len(intens)

    # Build intens string
    inten_str = ''
    for i, inten in enumerate(intens):
        if ((i+1) % 6) == 0 and (i+1) != len(intens):
            inten_str += '{0:<8.1f}\n'.format(int(inten))
        else:
            inten_str += '{0:<8.1f}'.format(inten)

    # Indent the lines
    inten_str = indent(inten_str, 4)

    return nintens, inten_str


def format_rotor_key_defs(rotor_keyword_vals):
    """ Formats strings that contain the 'Group', 'Axis', and 'Symmetry'
        keywords and values that are used to define hindered rotors and
        internal rotors in MESS input files.

        :param rotor_keyword_vals: values for the for some rotor keyword
        :type: rotor_keyword_vals: list(int)
        :return rotor_keyword_str: MESS-format string containing values
        :rtype str
    """

    # Build string containing the values of each keyword
    rotor_keyword_str = ''
    for vals in rotor_keyword_vals:
        rotor_keyword_str += '{0:<4d}'.format(vals+1)

    return rotor_keyword_str


def format_rotor_potential(potential):
    """ Formats the potential energy surface along a rotor into a string
        used to define hindered rotors and internal rotors in MESS input files.

        :param potential: value of the potential along torsion (kcal.mol-1)
        :type potential: list(float)
        :return npotential: number of values in the potential
        :rtype int
        :return potential_str: values of potential in a MESS-format string
        :rtype str
    """

    # Get the number of the terms in the potential
    npotential = len(potential)

    # Build potentials string
    potential_str = ''
    for i, energy in enumerate(potential.values()):
        if ((i+1) % 6) == 0 and (i+1) != npotential:
            potential_str += '{0:<8.2f}\n'.format(energy)
        else:
            potential_str += '{0:<8.2f}'.format(energy)

    # Indent the lines
    potential_str = indent(potential_str, 4)

    return npotential, potential_str


def format_rovib_coups(rovib_coups):
    """ Formats the matrix of rovibrational coupling terms for a species
        into a string appropriate for a MESS input file.

        :param rovib_coups: rovibrational coupling matrix
        :type rovib_coups: numpy.ndarray
        :return rovib_coups_str: values of potential in a MESS-format string
        :rtype str
    """

    # Join the values into a string
    rovib_coups_str = '  '.join(str(val) for val in rovib_coups)

    # Indent the lines
    rovib_coups_str = indent(rovib_coups_str, 4)

    return rovib_coups_str


def format_rot_dist_consts(rot_dists):
    """ Formats the list of rotational distortion constants
        into a string appropriate for a MESS input file.

        :param rot_dists: rotational distortion constants: [['aaa'], [val]]
        :type rot_dists: list(list(str), list(float))
        :return rot_dists_str: values of potential in a MESS-format string
        :rtype string
    """

    # Build rotational dists string
    rot_dists_str = ''
    for i, const in enumerate(rot_dists):
        rot_dists_str += '  '.join(map(str, const))
        if (i+1) != len(rot_dists):
            rot_dists_str += '\n'

    # Indent the lines
    rot_dists_str = indent(rot_dists_str, 4)

    return rot_dists_str


def format_xmat(xmat):
    """ Formats the anharmonicity (X) matrix for a species
        into a string appropriate for a MESS input file.

        :param xmat: anharmonicity matrix (cm-1)
        :type xmat: list(list(float))
        :return xmat_str: anharmonicity matrix in a MESS-format string
        :rtype string
    """

    xmat = numpy.array(xmat)

    # Loop over the rows of the anharm numpy array
    xmat_str = ''
    for i in range(xmat.shape[0]):
        xmat_str += ' '.join(
            ['{0:>12.5f}'.format(val) for val in list(xmat[i, :i+1])
             if val != 0.0]
        )
        if (i+1) != xmat.shape[0]:
            xmat_str += '\n'

    # Indent the lines
    xmat_str = indent(xmat_str, 2)

    return xmat_str


def molec_spec_format(geo):
    """ Parses out the atom labels of a Cartesian geometry and
        formats them into a string appropriate for definining
        molecular species for Monte Carlo calculations in MESS.

        :param geo: geometry
        :type geo: list
        :return atom_lst_str
        :rtype: string
    """

    # Build geom string; converting the coordinates to angstrom
    atom_lst_str = ''
    for (asymb, _) in geo:
        atom_lst_str += '{:s} '.format(asymb)

    # Remove final newline character
    atom_lst_str = atom_lst_str.rstrip()

    # Indent the lines
    atom_lst_str = indent(atom_lst_str, 6)

    return atom_lst_str


def format_flux_mode_indices(atom_idxs):
    """ Formates the atom indices into a string that is used
        to define the fluxional (torsional) modes of a
        molecular species for Monte Carlo calculations in MESS.

        :param atom_idxs: idxs of atoms involved in fluxional mode
        :type atom_idxs: list(int)
        :return flux_mode_idx_str: formatted string of indices
        :rtype: string
    """

    # Build string containing the values of each keyword
    flux_mode_idx_str = ''
    for vals in atom_idxs:
        flux_mode_idx_str += '{0:<4d}'.format(vals)

    return flux_mode_idx_str


# Helpful checker to set MESS string writing
def is_atom_in_str(spc_str):
    """ Checks a MESS-formatted species data string to see
        if the species is, or contains, an Atom species definition.

        :param: spc_str: MESS species string
        :type spc_str: str
        rtype: bool
    """
    return bool('Atom' in spc_str)
