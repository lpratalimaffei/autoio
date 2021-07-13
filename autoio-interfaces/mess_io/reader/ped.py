"""
    Read the product energy distribution and store the distributions obtained
"""
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from mess_io.reader import util
from mess_io.reader import statmodels


def ped_names(input_str):
    """ Reads the PEDSpecies and PEDOutput from the MESS input file string
        that were used in the master-equation calculation.
        :param input_str: string of lines of MESS input file
        :type input_str: str
        :return PEDSpecies: list of names of connected REACS/PRODS
        :rtype: list(str)
        :return PEDOutput: output file with the product energy distribution
        :rtype: str
    """

    # Get the MESS input lines
    mess_lines = input_str.splitlines()
    check = 0
    PEDSpecies = None
    PEDOutput = None
    for line in mess_lines:
        if 'PEDSpecies' in line:
            check += 1
            PEDSpecies = line.strip().split()[1].split('_')
        if 'PEDOutput' in line:
            check += 1
            PEDOutput = line.strip().split()[1]
        if check == 2:
            # found everything you need: exit from loop
            break

    if not PEDSpecies or not PEDOutput:
        print('Error: PEDSpecies and PEDOutput options incomplete. Exiting now')
        sys.exit()

    return PEDSpecies, PEDOutput


def get_ped(pedoutput_str, species, energies):
    """ Read the pedoutput file and extracts product energy distribution at T,P
        Energy in the output is set with respect to the ground energy of the products
        :param pedoutput_str: string of lines of PEDOutput file
        :type pedoutput_str: str
        :param species: species of interest in pedoutput
        :type species: list(str)
        :param energies: dictionary of energies of the species in the PES
        :type energies: dct(float)

        :return ped_df: dataframe(columns:P, rows:T) with the Series of energy distrib
        :rtype ped_df: dataframe(series(float))
    """
    species_string = species[0] + '->' + species[1]
    print(species_string)
    ped_lines = pedoutput_str.splitlines()
    # 0th of the energy: products energy
    E0 = energies[species[0]]-energies[species[1]]

    # find where data of interest are
    pressure_i = util.where_in('pressure', ped_lines)
    temperature_i = util.where_in('temperature', ped_lines)
    species_i = util.where_in(species_string, ped_lines)+1
    empty_i = util.where_is('', ped_lines)
    final_i = np.array([empty_i[i < empty_i][0] for i in species_i])

    # get T, P list
    pressure_lst = np.array([ped_lines[P].strip().split('=')[1]
                             for P in pressure_i], dtype=float)
    temperature_lst = np.array([ped_lines[T].strip().split('=')[1]
                                for T in temperature_i], dtype=float)
    # allocate empty dataframe
    ped_df = pd.DataFrame(index=list(set(temperature_lst)),
                          columns=list(set(pressure_lst)), dtype=object)
    # extract the data
    for i in np.arange(0, len(species_i)):
        P, T, i_in, i_fin = [pressure_lst[i],
                             temperature_lst[i], species_i[i], final_i[i]]
        energy, probability = np.array(
            [line.strip().split() for line in ped_lines[i_in:i_fin]], dtype=float).T
        energy = energy + E0  # rescale energy
        # build the series and put in dataframe after removing negative probs and renormalizing
        # integrate with the trapezoidal rule
        prob_en = pd.Series(probability, index=energy, dtype=float)
        if len(prob_en[prob_en < 0]) > 0:
            # if there are negative values of the probability: remove them
            prob_en = prob_en[:prob_en[prob_en < 0].index[0]]

        prob_en = prob_en.iloc[:-1].sort_index()  # skip last value
        # delta_E = [abs(DE) for DE in [prob_en.index[1:] - prob_en.index[:-1]]]
        # norm_factor = np.sum((prob_en.values[1:] + prob_en.values[:-1])*delta_E) / 2.
        # integrate with trapz
        norm_factor = np.trapz(prob_en.values, x=prob_en.index)
        ped_df.loc[T][P] = prob_en/norm_factor

    return ped_df


MW_dct_elements = {
    'C': 12e-3,
    'N': 14e-3,
    'O': 16e-3,
    'H': 1e-3,
    'S': 32e-3,
    'P': 31e-3,
    'F': 19e-3
}  # kg/mol


def ped_dof_MW(block):
    """ Gets the N of degrees of freedom and MW of each species
        :param block: bimol species of which you want the dofs
        :type block: list(str1, str2)
        :return dof_dct, rot_dct, MW_dct: dct with dofs and molecular weigths 
                    {prodi: Ni}
        :rtype: dataframe(index=species, columns=['vib dof', 'rot dof', 'mw'])
    """
    info_array = np.zeros((3, 3))
    keys = []
    atoms_ts = 0
    # extract N of dofs and MW
    for i, block_i in enumerate(block):
        info = block_i.splitlines()
        where_name = util.where_in('Species', info)[0]
        where_hind = util.where_in('Hindered', info)
        where_geom = util.where_in('Geometry', info)[0]
        N_atoms = int(info[where_geom].strip().split()[1])
        atoms_ts += N_atoms

        key = info[where_name].strip().split()[1]
        keys.append(key)
        try:
            where_freq = util.where_in('Frequencies', info)[0]
            N_dof = int(info[where_freq].strip().split()[1]) + len(where_hind)
            if 3*N_atoms - N_dof == 6:
                rot_dof = 3
            else:
                rot_dof = 2
        except IndexError:
            # if 1 atom only: no 'Frequencies', set to 0
            N_dof = 0
            rot_dof = 0
        # this allows to get 3N-5 or 3N-6 without analyzing the geometry
        info_array[i, 0] = N_dof
        info_array[i, 1] = rot_dof

        # MW from type of atoms:
        geom_in = where_geom+1
        geom_fin = geom_in+N_atoms
        atoms_array = np.array([geomline.strip().split()[0]
                                for geomline in info[geom_in:geom_fin]])

        mw = np.sum(np.array([MW_dct_elements[at]
                              for at in atoms_array], dtype=float))
        info_array[i, 2] = mw

    keys.append('TS')
    # assume there are no linear TSs
    info_array[2, :] = [3*atoms_ts - 7, 3, info_array[0, 2]+info_array[1, 2]]

    dof_info = pd.DataFrame(info_array, index=keys, columns=[
                            'vib dof', 'rot dof', 'mw'])
    print(dof_info)
    return dof_info


def ped_prod1(ped_df, prod1, modeltype, dos_df=None, dof_info=None, E_BW=None):
    """ call ped_models class in statmodels and compute P(E1)

        :param ped_df: dataframe(columns:P, rows:T) with the Series of energy distrib
        :type ped_df: dataframe(series(float))
        :param dos_df: rovibr dos for each fragment
        :type dos_df: dataframe(index=energy, columns=[frag1, frag2])
        :param dof_dct: dct with dofs {prodi: Ni}
        :type dof_dct: dct
        :param dof_dct: dct with molecular weights {prodi: MWi}
        :type dof_dct: dct
        :param prod1: fragment of the energy distribution we want
        :type prod1: str
        :param E_BW: backward energy barrier TS-PRODS
        :type E_BW: float
        :param modeltype: type of model to be implemented
        :type modeltype: str

        :return P_E1_prod1: energy distribution of the product prod
        :rtype: dataframe(series(float, index=energy), index=T, columns=P)
    """

    # call class
    ped_prod1_fct = statmodels.ped_models(
        ped_df, dos_df=dos_df, dof_info=dof_info, E_BW=E_BW, prod1=prod1)
    P_E1_prod1 = ped_prod1_fct.compute_ped(modeltype)

    return P_E1_prod1
