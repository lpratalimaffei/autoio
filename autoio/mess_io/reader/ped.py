"""
    Read the product energy distribution and store the distributions obtained
"""
import sys
import numpy as np
import pandas as pd
from autoio import mess_io
from autoio.mess_io.reader import util


def get_ped_names(input_str):
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


def get_ped_dof(input_str, species_name):
    """ Gets the N of degrees of freedom of each product in species
        :param input_str: string of lines of MESS input file
        :type input_str: str
        :param species_name: bimol species of which you want the dofs
        :type species_name: str
        :return dof_dct: dct with dofs {prodi: Ni}
        :rtype: dct
    """

    # get species blocks
    species_blocks = mess_io.reader.get_species(input_str)
    fragments_block = species_blocks[species_name]
    dof_dct = {}
    # extract N of dofs
    for block_i in fragments_block:
        info = block_i.splitlines()
        where_name = util.where_in('Species', info)[0]
        where_hind = util.where_in('Hindered', info)
        key = info[where_name].strip().split()[1]
        try:
            where_freq = util.where_in('Frequencies', info)[0]
            N_dof = int(info[where_freq].strip().split()[1]) + len(where_hind)
        except IndexError:
            # if 1 atom only: no 'Frequencies', set to 0
            N_dof = 0
        # this allows to get 3N-5 or 3N-6 without analyzing the geometry
        dof_dct[key] = N_dof

    return dof_dct

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
        energy = energy + E0 # rescale energy
        # build the series and put in dataframe after removing negative probs and renormalizing
        # integrate with the trapezoidal rule
        prob_en = pd.Series(probability, index=energy, dtype=float)
        prob_en = prob_en[:prob_en[prob_en < 0].index[0]]
        prob_en = prob_en.iloc[:-1] #skip last value
        # delta_E = [abs(DE) for DE in [prob_en.index[1:] - prob_en.index[:-1]]]
        # norm_factor = np.sum((prob_en.values[1:] + prob_en.values[:-1])*delta_E) / 2.
        # integrate with trapz: put minus in front cause energies have decreasing values seo DE <0 
        norm_factor = -np.trapz(prob_en.values, x=prob_en.index)
        ped_df.loc[T][P] = prob_en/norm_factor

    #E_units = ped_lines[species_i[0]].strip().split()[1]

    return ped_df


def prod_ped_equip(ped_df, dof_dct, prod):
    """ Derive the energy distribution of 1 product from the energy equipartition theorem

        :param ped_df: dataframe(columns:P, rows:T) with the Series of energy distrib
        :type ped_df: dataframe(series(float))
        :param dof_dct: dct with dofs {prodi: Ni}
        :type dof_dct: dct
        :param prod: fragment of the energy distribution we want
        :type prod: str
        :return ped_df_prod: energy distribution of the product prod
        :rtype: dataframe(series(float))
    """
    # derive the energy fraction from the equipartition theorem
    N_dof_prod = dof_dct[prod]
    dof_dct.pop(prod)
    N_dof_rest = list(dof_dct.values())[0]
    beta_prod = (N_dof_prod+3/2)/(N_dof_prod+N_dof_rest+9/2)
    # 3/2: 1/2kbT for each rotation, no trasl (ts trasl energy preserved)
    # 9/2: 1/2kbT*6 rotational dofs for products, +3 for relative trasl
    print('fraction of energy transferred to products: {:.2f}'.format(beta_prod))
    # rescale all energies with beta: allocate values in new dataframe
    ped_df_prod = pd.DataFrame(index=ped_df.index,
                        columns=ped_df.columns, dtype=object)
    for P in ped_df.columns:
        for T in ped_df.index:
            idx_new = ped_df[P][T].index *beta_prod
            vals = ped_df[P][T].values
            ped_df_prod[P][T] = pd.Series(vals, index=idx_new)

    return ped_df_prod




