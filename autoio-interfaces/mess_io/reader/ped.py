"""
    Read the product energy distribution and store the distributions obtained
"""
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from mess_io.reader import util


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
        :return dof_dct: dct with dofs {prodi: Ni}
        :rtype: dct
    """

    dof_dct = {}
    MW_dct = {}
    # extract N of dofs and MW
    for block_i in block:
        info = block_i.splitlines()
        where_name = util.where_in('Species', info)[0]
        where_hind = util.where_in('Hindered', info)
        where_geom = util.where_in('Geometry', info)[0]
        N_atoms = int(info[where_geom].strip().split()[1])

        key = info[where_name].strip().split()[1]
        try:
            where_freq = util.where_in('Frequencies', info)[0]
            N_dof = int(info[where_freq].strip().split()[1]) + len(where_hind)
        except IndexError:
            # if 1 atom only: no 'Frequencies', set to 0
            N_dof = 0
        # this allows to get 3N-5 or 3N-6 without analyzing the geometry
        dof_dct[key] = N_dof

        # MW from type of atoms:
        geom_in = where_geom+1
        geom_fin = geom_in+N_atoms
        atoms_array = np.array([geomline.strip().split()[0]
                                for geomline in info[geom_in:geom_fin]])

        MW_dct[key] = np.sum(np.array([MW_dct_elements[at]
                                       for at in atoms_array], dtype=float))

    return dof_dct, MW_dct


def dos_trasl(m1, m2, E_grid, P, T):
    """ Compute the translational density of states per unit volume
        m1, m2: MW in kg/mol
        E_grid: energy grid in kcal/mol
        P, T: pressure and temperature in SI units [Pa, K]
    """
    # conversions
    NAVO = 6.022e+23  # molec/mol
    E_grid_J = E_grid*4184/NAVO  # kcal/mol*J/kcal/NAVO=J
    m1 /= NAVO  # kg/mol/(molecule/mol)
    m2 /= NAVO
    red_mass = (m1 * m2) / (m1 + m2)    # kg
    h = 6.626e-34   # Planck's constant, J*s

    # rhotr/V = pi/4*(8m/h2)^3/2*e^1/2
    rho = (np.pi/4*np.power(8*red_mass/h**2, 3/2) *
           np.power(E_grid_J, 1/2))  # unit: 1/m3/J

    # consider the molar volume kBT/P [m3/molec]
    V_mol = 1.38e-23*T/P  # m3/molec
    rho_kcalmol = rho*V_mol*NAVO*4184  # 1/m3*(m3/molec)*(molec/mol)/J*(J/kcal)

    dos_tr_series = pd.Series(rho, index=E_grid)

    return dos_tr_series


def prod_ped_dos(ped_df, dos_df, MW_dct, prod1):
    """ Derive the energy distribution of 1 product from the 
        convolution of the density of states

        :param ped_df: dataframe(columns:P, rows:T) with the Series of energy distrib
        :type ped_df: dataframe(series(float))
        :param dos_df: rovibr dos for each fragment
        :type dos_df: dataframe()
        :param prod1: fragment of the energy distribution we want
        :type prod1: str
        :return ped_df_prod: probability distribution of the energy of prod1
        :rtype: dataframe(series(float))
    """

    def convolve_rho_E(rho1, rho2):
        """ convolution of 2 rhovibrational density of states sharing the same energy vector
            :param rho1, rho2: density of states
            :type rho1, rho2: Series [index=energy kcal/mol, value=density mol/kcal]
            :return rho_conv: convoluted dos rho_conv(E)=int(rho1(E')*rho2(E-E')dE')
            :rtype: Series [index=energy kcal/mol, value=density mol/kcal]
        """

        if any(rho1.index != rho2.index):
            raise Exception('The energy grids do not match.')
            exit()

        Etot = rho1.index  # index is the variable in which you want to integrate
        rho_conv = pd.Series(index=Etot, dtype=float)

        for e in Etot:
            Eint = Etot[Etot <= e]  # energia per integrazione
            try:
                e_minus_Eint = e-Eint
                # round necessary to avoid numerical differences in float precision
                integrand = rho1[Eint] * rho2[e_minus_Eint]
            except KeyError:
                e_minus_Eint = np.array([round(diff, 2) for diff in e-Eint])  # e-Eint
                # round necessary to avoid numerical differences in float precision
                integrand = rho1[Eint] * rho2[e_minus_Eint]
            rho_conv[e] = np.trapz(integrand, x=Eint)

        return rho_conv

    # preallocations
    ped_df_prod = pd.DataFrame(index=ped_df.index,
                               columns=ped_df.columns, dtype=object)

    # dos_df: estrai le densità di stati
    if 0 not in dos_df.index:
        df_zero = pd.DataFrame(np.zeros((1,2), dtype=float), index=[
                            0.0], columns=dos_df.columns)
        dos_df = pd.concat([df_zero, dos_df])
    print(dos_df)
    E_dos0 = dos_df.index
    prod2 = dos_df.columns[dos_df.columns != prod1][0]

    for P in ped_df.columns:
        for T in ped_df.sort_index().index:
            E = ped_df[P][T].index
            # riduci valori di E, E_dos al range minimo
            upp_bound = min(E_dos0[-1], E[-1])
            E_dos = E_dos0[E_dos0 <= upp_bound]
            E = E[E <= upp_bound]
            # estraggo ped, rhovib, rhotrasl
            ped_E = ped_df[P][T][E].values
            f_ped_E = interp1d(E, ped_E, kind='cubic',
                               fill_value='extrapolate')
            # technically the "extrapolate" shouldn't be necessary
            rho_rovib_prod1 = dos_df[prod1][E_dos]  # series
            rho_rovib_prod2 = dos_df[prod2][E_dos]  # series
            rho_trasl = dos_trasl(
                MW_dct[prod1], MW_dct[prod2], E_dos, P*101325, T)
            E1_vect = np.copy(E_dos)
            P_E1 = pd.Series(index=E1_vect[1:-1], dtype=float)
            # calculate rho_non1 as a function of the energy
            # (the vector E-E1 will be selected later)
            rho_non1 = convolve_rho_E(rho_rovib_prod2, rho_trasl)

            for E1 in E1_vect[1:-1]:
                rho1_E1 = rho_rovib_prod1[E1]
                # sto facendo cicli su E1, quindi l'en totale deve essere almeno quanto E1
                Eint = E_dos[E_dos >= E1]
                # Eint perché è quella che usi nell'integrale
                try:
                    E_minus_E1 = Eint-E1
                    rho_non1_E_minus_E1 = rho_non1[E_minus_E1].values
                    # prova2 considero solo rovibr
                    # rho_non1_E_minus_E1 = rho_rovib_prod2[E_minus_E1].values
                except KeyError:
                    E_minus_E1 = np.array([round(diff, 2) for diff in Eint-E1])
                    rho_non1_E_minus_E1 = rho_non1[E_minus_E1].values
                    # prova2
                    # rho_non1_E_minus_E1 = rho_rovib_prod2[E_minus_E1].values
                # put it as an exception cause the code gets much slower
                # compute the convolution between rho1 and rhonon1
                # int(rho1(E1)*rho_non1(E-E1)dE)
                rho1_rhonon1 = rho1_E1*rho_non1_E_minus_E1
                norm_factor_P_E1Etot = np.trapz(rho1_rhonon1, x=Eint)
                P_E1Etot = rho1_rhonon1/norm_factor_P_E1Etot

                # compute P(E1): you need to derive Pped[Eint]
                P_ped = f_ped_E(Eint)
                P_E1Etot_P_ped = P_E1Etot*P_ped
                P_E1[E1] = np.trapz(P_E1Etot_P_ped, x=Eint)
            
            norm_factor_P_E1 = np.trapz(P_E1[E1_vect[1:-1]].values, x=E1_vect[1:-1])
            P_E1_norm = P_E1/norm_factor_P_E1
            ped_df_prod[P][T] = P_E1_norm
            print(P, T, max(E1_vect[1:-1]), P_E1_norm.idxmax(), P_E1_norm.max(), '\n')
            #print(np.trapz(ped_df_prod[P][T], x=E1_vect[1:-1]))
            
    return ped_df_prod


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
    print('dof of the fragment {}: {}'.format(prod, N_dof_prod))
    dof_dct.pop(prod)
    N_dof_rest = list(dof_dct.values())[0]
    beta_prod = (N_dof_prod+3/2)/(N_dof_prod+N_dof_rest+9/2)
    # 3/2: 1/2kbT for each rotation, no trasl (ts trasl energy preserved)
    # 9/2: 1/2kbT*6 rotational dofs for products, +3 for relative trasl
    print(
        'fraction of energy transferred to products: {:.2f}'.format(beta_prod))
    # rescale all energies with beta: allocate values in new dataframe
    ped_df_prod = pd.DataFrame(index=ped_df.index,
                               columns=ped_df.columns, dtype=object)
    for P in ped_df.columns:
        for T in ped_df.sort_index().index:
            idx_new = ped_df[P][T].index * beta_prod
            norm_factor = np.trapz(ped_df[P][T].values, x=idx_new)
            vals = ped_df[P][T].values/norm_factor
            ped_df_prod[P][T] = pd.Series(vals, index=idx_new)
            print(P, T, max(idx_new), ped_df_prod[P][T].idxmax(), ped_df_prod[P][T].max(), '\n')
    print('\n\n\n')
    return ped_df_prod


######################### FROM LEI LEI'S CODE ########################
def rho_vib_E(freq_ls, energy_step, max_energy, initial_value=None, calc_step=5):
    '''Calculate the vibrational density of states using Beyer-Swinehart direct count algorithm.
       Inputs are in order the vibrational frequencies of the harmonic oscillator, the energy step
       and the maximum energy for rovibrational density of states output, the initial vectors for
       direct count calculation, which should be the energy vector and density of states vector of
       the rotational density of states, and the energy step for direct count.
       Vibrational frequencies and energy are in the unit of cm-1.
       Output is the density of states at specific energy levels in the unit of 1/cm-1.'''

    # initialization

    E_grid = np.array(range(0, int(max_energy), int(
        energy_step)) + [max_energy])    # outpyt grids
    rho = np.zeros(len(E_grid[1:]))  # zero energy is not counted

    # main loop
    if initial_value == None:
        calc_grid = np.array(range(0, int(max_energy), int(
            calc_step)) + [max_energy])                # calculation grids
        calc_rho = np.zeros(int(max_energy / calc_step) + 1)
        calc_rho[0] = 1.
    else:
        calc_grid = initial_value[0]
        calc_rho = initial_value[1]
    # direct count
    for f in freq_ls:
        pos = np.sum(calc_grid < round(f))
        for e in calc_grid[pos:]:
            curr_loc = np.sum(calc_grid < e)
            pri_loc = np.sum(calc_grid < e-round(f))
            calc_rho[curr_loc] += calc_rho[pri_loc]

    # output
    for n, e in enumerate(E_grid[1:]):
        if n == 0:
            pos = 0
        pre_pos = pos
        pos = np.sum(calc_grid <= e)
        rho[n] = np.sum(calc_rho[pre_pos:pos]) / energy_step

    return E_grid[1:], rho


def rho_rot_E(one_D_B_ls, two_D_B_ls, one_D_sigma_ls, two_D_sigma_ls, energy_step, max_energy):
    '''Calculate the rotational density of states for molecules that are unhindered.
       Inputs are in order rotational constants for 1D and 2D rotors, the symmetry
       number for 1D and 2D rotors, the calculation energy step and the maximum energy
       to be considered. Rotational constants and energy are in the unit of cm-1.
       Output is the density of state at specified energy levels in the unit of 1/cm-1.'''
    # initialization
    E_grid = np.array(range(0, int(max_energy), int(energy_step))[
                      1:] + [max_energy])    # outpyt grids
    rho = np.zeros(len(E_grid))

    for n, e in enumerate(E_grid):
        if len(one_D_B_ls) == 0:
            temp = np.prod([np.power(B_i * sigma_i, -1)
                            for (B_i, sigma_i) in zip(two_D_B_ls, two_D_sigma_ls)])
        elif len(two_D_B_ls) == 0:
            temp = np.prod([np.power(B_i, -0.5) / sigma_i for (B_i,
                                                               sigma_i) in zip(one_D_B_ls, one_D_sigma_ls)])
        else:
            temp_1 = np.prod([np.power(
                B_i, -0.5) / sigma_i for (B_i, sigma_i) in zip(one_D_B_ls, one_D_sigma_ls)])
            temp_2 = np.prod([np.power(B_i * sigma_i, -1.)
                              for (B_i, sigma_i) in zip(two_D_B_ls, two_D_sigma_ls)])
            temp = temp_1 * temp_2

        rho[n] = np.power(np.pi, len(one_D_B_ls) / 2.) / \
            gamma(len(two_D_B_ls) + len(one_D_B_ls) / 2.)
        rho[n] *= np.power(e, len(two_D_B_ls) +
                           len(one_D_B_ls) / 2. - 1.) * temp

    return E_grid, rho


def convolve_rho_E(mode_1, mode_2):
    '''Convolve the rovibrational density of states with relative translational density of
       states to get the total density of states. Inputs are nested lists containing the
       energy levels and density of states for the modes to be convolved. To get a better
       result, it is optimal to have the same resolution for both modes. Output is the
       convolved energy levels and density of states in the unit of 1/cm-1.'''

    if any(mode_1[0] != mode_2[0]):
        raise Exception('The energy grids do not match.')
    E_grid = mode_1[0]
    delta_E = np.diff(E_grid)[0]
    rho_1 = np.insert(mode_1[1], 0, 1.)
    rho_2 = np.insert(mode_2[1], 0, 1.)
    rho = np.zeros(len(E_grid))

    for n, e in enumerate(E_grid):
        pos = np.sum(E_grid <= e) + 1
        temp = rho_1[:pos] * rho_2[:pos][::-1]
        rho[n] = np.sum(temp[1:] + temp[:-1]) / 2. * delta_E

    return E_grid, rho
