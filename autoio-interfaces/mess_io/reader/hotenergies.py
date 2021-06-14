"""
    Read the HotEnergies distribution and store it
"""

import numpy as np
import pandas as pd
from mess_io.reader import util


def get_hot_names(input_str):
    """ Reads the HotSpecies from the MESS input file string
        that were used in the master-equation calculation.
        :param input_str: string of lines of MESS input file
        :type input_str: str
        :return hotspecies: list of hotspecies
        :rtype: list(str)
    """

    # Get the MESS input lines
    mess_lines = input_str.splitlines()
    hotsp_i = util.where_in('HotEnergies', mess_lines)[0]
    N_hotsp = int(mess_lines[hotsp_i].strip().split()[1])
    hotspecies = [None]*N_hotsp

    for i, line in enumerate(mess_lines[hotsp_i+1:hotsp_i+1+N_hotsp]):
        hotspecies[i] = line.strip().split()[0]

    return hotspecies


def extract_hot_branching(hotenergies_str, hotspecies_lst, species_lst, T_lst, P_lst):
    """ Extract hot branching fractions for a single species
        :param hotenergies_str: string of mess log file
        :type hotenergies_str: str
        :param hotspecies_lst: list of hot species
        :type hotspecies_lst: list
        :param species_lst: list of all species on the PES
        :type species_lst: list
        :return hoten_dct: hot branching fractions for hotspecies
        :rtype hoten_dct: dct{hotspecies: df[P][T]:df[allspecies][energies]}
    """
    lines = hotenergies_str.splitlines()
    # for each species: dataframe of dataframes BF[Ti][pi]
    # each of them has BF[energy][species]
    # preallocations
    hoten_dct = {s: pd.DataFrame(index=T_lst, columns=P_lst)
                 for s in hotspecies_lst}
    # 1. for each P,T: extract the block
    PT_i_array = util.where_in(['Pressure', 'Temperature'], lines)
    Hot_i_array = util.where_in(['Hot distribution branching ratios'], lines)
    end_Hot_i_array = util.where_in(
        ['MasterEquation', 'method', 'done'], lines)
    # 2. find Hot distribution branching ratios:
    for i, Hot_i in enumerate(Hot_i_array):

        # extract block, PT, and species for which BF is assigned
        lines_block = lines[Hot_i+2:end_Hot_i_array[i]]

        P, T = [float(var)
                for var in lines[PT_i_array[i]].strip().split()[2:7:4]]

        species_BF_i = lines[Hot_i+1].strip().split()[3:]

        # for each hotspecies: read BFs
        for hotspecies in hotspecies_lst:
            hot_e_lvl, branch_ratio = [], []
            sp_i = util.where_in(hotspecies, species_BF_i)

            for line in lines_block:
                line = line.strip()
                if line.startswith(hotspecies):
                    branch_ratio_arr = np.array(
                        [x for x in line.split()[2:]], dtype=float)

                    # check that the value of the reactant branching is not negative
                    # if > 1, keep it so you can account for that anyway
                    if sp_i.size > 0:
                        if branch_ratio_arr[sp_i] < 0:
                            continue
                        elif branch_ratio_arr[sp_i] > 1:
                            branch_ratio_arr[sp_i] = 1
                    # remove negative values or values >1
                    br_filter = np.array(
                        [abs(x*int(x > 1e-5 and x <= 1)) for x in branch_ratio_arr], dtype=float)
                    # if all invalid: do not save
                    if all(br_filter == 0):
                        continue
                    br_renorm = br_filter/np.sum(br_filter)
                    # append values
                    branch_ratio.append(br_renorm)
                    hot_e_lvl.append(float(line.split()[1]))

            hot_e_lvl = np.array(hot_e_lvl)
            branch_ratio = np.array(branch_ratio)

    # 3. allocate in the dataframe
            bf_hotspecies = pd.DataFrame(
                0, index=hot_e_lvl, columns=species_lst)
            bf_hotspecies[species_BF_i] = branch_ratio
            hoten_dct[hotspecies][P][T] = bf_hotspecies
            # print(hoten_dct)
    return hoten_dct
