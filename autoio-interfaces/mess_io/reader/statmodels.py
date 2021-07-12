"""
    Read the product energy distribution and store the distributions obtained
"""
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from mess_io.reader import util


def dos_trasl(m1, m2, E_grid, P, T):
    """ Compute the translational density of states per unit volume
        m1, m2: MW in kg/mol
        E_grid: energy grid in kcal/mol (array)
        P, T: pressure and temperature in SI units [Pa, K]
        :return dos_tr_series: dos in mol/kcal
        :rtype: pd.Series(dos, index=energy)
    """
    # conversions
    NAVO = 6.022e+23  # molec/mol
    E_grid_J = E_grid*4184/NAVO  # kcal/mol*J/kcal/NAVO=J
    m1 /= NAVO  # kg/mol/(molecule/mol)
    m2 /= NAVO
    red_mass = (m1 * m2) / (m1 + m2)    # kg
    h = 6.626e-34   # Planck's constant, J*s
    # 1 molecule
    # rhotr/V = pi/4*(8m/h2)^3/2*e^1/2
    rho = (np.pi/4*np.power(8*red_mass/h**2, 3/2) *
           np.power(E_grid_J, 1/2))  # unit: 1/m3/J

    # consider the molar volume RT/P [m3/1]
    V_mol = 1.38e-23*T/P  # m3/1 for ideal gases
    # 1/m3*(m3/1)/J*(J/kcal)*(mol) = mol/kcal
    rho_kcal_mol = rho*V_mol*4184/NAVO

    dos_tr_series = pd.Series(rho_kcal_mol, index=E_grid)

    return dos_tr_series


def convolve_f(f1, f2, vect):
    """ convolve 2 functions f1 and f2; Integration vector is vect
        :param f1, f2: functions for convolution
        :type f1, f2: functions
        :param vect: vector for convolution
        :vect type: list or numpy array
        :return int_conv: convoluted integral int_conv(vect)=int(rho1(x)*rho2(vect-x)dx)
        :rtype: array
    """

    int_conv = []
    for up in vect:
        xint = np.array(vect[vect <= up])  # energia per integrazione
        up_minus_xint = up-xint
        integrand = f1(xint) * f2(up_minus_xint)
        int_conv.append(np.trapz(integrand, x=xint))

    int_conv = np.array(int_conv)

    return int_conv


class ped_models:

    def __init__(self, ped_df, prod1, dos_df=None, mw_dct=None, dof_dct=None, E_BW=None):
        """ initialize variables
            :param ped_df: dataframe(columns:P, rows:T) with the Series of energy distrib
            :type ped_df: dataframe(series(float))
            :param dos_df: rovibr dos for each fragment
            :type dos_df: dataframe()
            :param dof_dct: dct with dofs {prodi: Ni}
            :type dof_dct: dct
            :param prod1: fragment of the energy distribution we want
            :type prod1: str
        """
        self.ped_df = ped_df
        self.dos_df = dos_df
        self.mw_dct = mw_dct
        self.dof_dct = dof_dct
        self.E_BW = E_BW
        self.prod1 = prod1
        self.prod2 = self.dos_df.columns[self.dos_df.columns != self.prod1][0]
        self.N_dof_prod = dof_dct[self.prod1]
        self.N_dof_prod2 = dof_dct[self.prod2]
        self.N_dof_TS = dof_dct['TS']

        self.models_dct = {
            'equip_simple': self.equip_simple,
            'equip_phi': self.equip_phi,
            'rovib_dos': self.rovib_dos,
            'beta_phi1a': self.beta_phi1a,
            'beta_phi2a': self.beta_phi2a,
            'beta_phi3a': self.beta_phi3a
        }

    def compute_ped(self, modeltype):
        """ compute ped according to the desired model
        """

        try:
            ped_df_prod = self.models_dct[modeltype]()
            return ped_df_prod

        except KeyError:
            print('*Error: model not available. Please select among {} \n'.format(
                '\n'.join(self.models_dct.keys())))
            exit()

    def equip_simple(self):
        """ Derive the energy distribution of 1 product from the energy equipartition theorem

            :return ped_df_prod: energy distribution of the product prod
            :rtype: dataframe(series(float))
        """
        if self.N_dof_prod == None or self.N_dof_prod2 == None:
            print('Error: DOFs not defined, now exiting\n')
            sys.exit()

        # derive the energy fraction from the equipartition theorem
        print('dof of the fragment {}: {}'.format(self.prod1, self.N_dof_prod))
        beta_prod = (self.N_dof_prod+3/2) / \
            (self.N_dof_prod+self.N_dof_prod2+9/2)
        # 3/2: 1/2kbT for each rotation, no trasl (ts trasl energy preserved)
        # 9/2: 1/2kbT*6 rotational dofs for products, +3 for relative trasl
        print(
            'fraction of energy transferred to products: {:.2f}'.format(beta_prod))
        # rescale all energies with beta: allocate values in new dataframe
        ped_df_prod = pd.DataFrame(index=self.ped_df.index,
                                   columns=self.ped_df.columns, dtype=object)
        for P in self.ped_df.columns:
            for T in self.ped_df.sort_index().index:
                idx_new = self.ped_df[P][T].index * beta_prod
                norm_factor = np.trapz(self.ped_df[P][T].values, x=idx_new)
                vals = self.ped_df[P][T].values/norm_factor
                ped_df_prod[P][T] = pd.Series(vals, index=idx_new)
                print(P, T, max(idx_new), ped_df_prod[P][T].idxmax(
                ), ped_df_prod[P][T].max(), '\n')
        print('\n\n\n')

        return ped_df_prod

    def P_E1_phi(self, phi):
        """ Derive the energy distribution of 1 product from one
            of the statistical models by Danilack Goldsmith PROCI 2020
            phi is the average fraction of energy transferred to the products

            calculate P(E') = probability of fragment 1 to have energy E'
            a. alfa(E) = PED(E)
            b. select E' and calculate P(E',E) according to normal distribution
            c. integrate over dE: int(P(E',E)*PED(E)dE) = P(E')
        """

        def norm_distr(E1, E, phi, E_BW):
            """ P(E1, E) = exp(-(E1-phi*E)^2/(2^0.5*sigma(E_BW)))/(2*pi)^0.5/sigma(E_BW)
                mi = phi*E
                sigma = f(E_BW, phi*E)
                E1 is a number
                E is a vector
            """

            mi = np.array(phi*E, dtype=float)
            # correlation from Danilack Goldsmith - I add the additional fraction of energy transferred to the products
            sigma = np.array(0.87+0.04*(E_BW+phi*(E-E_BW)), dtype=float)
            # sigma = 0.87+0.04*E_BW
            num = np.exp(-((E1-mi)/(2**0.5)/sigma)**2)
            den = np.power(2*np.pi, 0.5)*sigma

            P_E1E = num/den

            # norm.pdf(E1, mi_e, sigma_e)

            return P_E1E

        if self.E_BW == None:
            print('Error: backward energy barrier not defined, exiting now\n')
            sys.exit()

        # preallocations
        ped_df_prod = pd.DataFrame(index=self.ped_df.index,
                                   columns=self.ped_df.columns, dtype=object)

        Emax = max(self.ped_df[self.ped_df.columns[-1]]
                   [self.ped_df.sort_index().index[-1]].index)
        # just to save stuff, remove later

        for P in self.ped_df.columns:

            P_E1_df = pd.DataFrame(
                index=np.arange(0.0, round(Emax, 1), 0.1).round(
                    decimals=1), columns=self.ped_df.sort_index().index, dtype=float)
            # just to save stuff, remove later

            for T in self.ped_df.sort_index().index:
                E = self.ped_df[P][T].index
                ped_E = self.ped_df[P][T][E].values
                f_ped_E = interp1d(E, ped_E, kind='cubic',
                                   fill_value='extrapolate')
                E = np.arange(0.0, round(max(E), 1), 0.1).round(
                    decimals=1)  # default step size 0.1
                P_E1 = pd.Series(index=E[1:-1], dtype=float)

                for E1 in E[1:-1]:

                    P_E1E = norm_distr(E1, E[1:-1], phi, self.E_BW)
                    # compute P(E1): you need to derive Pped[Eint]
                    P_ped = f_ped_E(E[1:-1])
                    P_E1Etot_P_ped = P_E1E*P_ped
                    P_E1[E1] = np.trapz(P_E1Etot_P_ped, E[1:-1])

                norm_factor_P_E1 = np.trapz(
                    P_E1[E[1:-1]].values, x=E[1:-1])
                P_E1_norm = P_E1/norm_factor_P_E1
                ped_df_prod[P][T] = P_E1_norm

                P_E1_df[T][E[1:-1]] = P_E1_norm

                print(P, T, max(E[1:-1]),
                      P_E1_norm.idxmax(), P_E1_norm.max(), '\n')
                #print(np.trapz(ped_df_prod[P][T], x=E1_vect[1:-1]))

            # write file - remove later
            P_E1_df = P_E1_df.reset_index()
            header_label = np.array(P_E1_df.columns, dtype=str)
            header_label[0] = 'E [kcal/mol]'
            labels = '\t\t'.join(header_label)
            np.savetxt('PE1_{}.txt'.format(P), P_E1_df.values,
                       delimiter='\t', header=labels, fmt='%1.2e')

        return ped_df_prod

    def equip_phi(self):
        """ Derive the energy distribution of 1 product from the energy equipartition theorem

            :return ped_df_prod: energy distribution of the product prod
            :rtype: dataframe(series(float))
        """
        if self.N_dof_prod == None or self.N_dof_prod2 == None:
            print('Error: DOFs not defined, now exiting\n')
            sys.exit()

        # derive the energy fraction from the equipartition theorem
        print('dof of the fragment {}: {}'.format(self.prod1, self.N_dof_prod))
        phi_prod = (self.N_dof_prod+3/2) / \
            (self.N_dof_prod+self.N_dof_prod2+9/2)

        ped_df_prod = self.P_E1_phi(phi_prod)

        return ped_df_prod

    def beta_phi1a(self):
        """ Derive the energy distribution of 1 product from 
            statistical model phi1a Danilack Goldsmith PROCI 2020

            :return ped_df_prod: energy distribution of the product prod
            :rtype: dataframe(series(float))
        """

        # derive the energy fraction phi
        phi1a = self.N_dof_prod/self.N_dof_TS
        print(
            'fraction of energy transferred to products phi1a: {:.2f}'.format(phi1a))
        # rescale all energies with beta: allocate values in new dataframe
        ped_df_prod = self.P_E1_phi(phi1a)

        return ped_df_prod

    def beta_phi2a(self):
        """ Derive the energy distribution of 1 product from 
            statistical model phi2a Danilack Goldsmith PROCI 2020

            :return ped_df_prod: energy distribution of the product prod
            :rtype: dataframe(series(float))
        """
        # derive the energy fraction phi
        phi2a = (self.N_dof_prod+3)/(self.N_dof_TS+3)
        print(
            'fraction of energy transferred to products phi2a: {:.2f}'.format(phi2a))
        ped_df_prod = self.P_E1_phi(phi2a)

        return ped_df_prod

    def beta_phi3a(self):
        """ Derive the energy distribution of 1 product from 
            statistical model phi2a Danilack Goldsmith PROCI 2020

            :return ped_df_prod: energy distribution of the product prod
            :rtype: dataframe(series(float))
        """
        # derive the energy fraction phi

        phi3a = (self.N_dof_prod+3)/(self.N_dof_prod+self.N_dof_prod2+9)
        print(
            'fraction of energy transferred to products phi3a: {:.2f}'.format(phi3a))
        ped_df_prod = self.P_E1_phi(phi3a)

        return ped_df_prod

    def rovib_dos(self):
        """ Derive the energy distribution of 1 product from the 
            convolution of the density of states

            :return ped_df_prod: probability distribution of the energy of prod1
            :rtype: dataframe(series(float))
        """
        # checks on input
        try:
            self.dos_df.empty
        except AttributeError:
            print('*Error: dos not defined, exiting now \n')
            sys.exit()
        if self.mw_dct == None:
            print('*Error: mw not defined, exiting now \n')
            sys.exit()

        # preallocations
        ped_df_prod = pd.DataFrame(index=self.ped_df.index,
                                   columns=self.ped_df.columns, dtype=object)

        # dos_df: extract density of states
        if 0 not in self.dos_df.index:
            df_zero = pd.DataFrame(np.zeros((1, 2), dtype=float), index=[
                0.0], columns=self.dos_df.columns)
            self.dos_df = pd.concat([df_zero, self.dos_df])

        E_dos0 = self.dos_df.index

        # dos functions for prod1, prod2
        f_rho_rovib_prod1 = interp1d(E_dos0, self.dos_df[self.prod1][E_dos0].values, kind='cubic',
                                     fill_value='extrapolate')
        f_rho_rovib_prod2 = interp1d(E_dos0, self.dos_df[self.prod2][E_dos0].values, kind='cubic',
                                     fill_value='extrapolate')

        Emax = max(self.ped_df[self.ped_df.columns[-1]]
                   [self.ped_df.sort_index().index[-1]].index)
                   
        Evect_full = np.linspace(0, round(Emax, 1), num=(round(Emax-0)/0.1))
        for P in self.ped_df.columns:

            P_E1_df = pd.DataFrame(
                index=Evect_full[1:], columns=self.ped_df.sort_index().index, dtype=float)
            # just to save stuff, remove later

            for T in self.ped_df.sort_index().index:
                E = self.ped_df[P][T].index
                ped_E = self.ped_df[P][T][E].values
                f_ped_E = interp1d(E, ped_E, kind='cubic',
                                   fill_value='extrapolate')
                # riduci valori di E, E_dos al range minimo
                upp_bound = min(E_dos0[-1], E[-1])
                # default step size 0.1
                E = Evect_full[Evect_full <= upp_bound]
                # estraggo ped, rhovib, rhotrasl
                # technically the "extrapolate" shouldn't be necessary
                rho_trasl = dos_trasl(
                    self.mw_dct[self.prod1], self.mw_dct[self.prod2], E, P*101325, T)
                f_rhotrasl = interp1d(E, dos_trasl(
                    self.mw_dct[self.prod1], self.mw_dct[self.prod2], E, P*101325, T).values,
                    kind='cubic', fill_value='extrapolate')

                P_E1E = pd.DataFrame(index=E, columns=E[3:], dtype=float)
                P_E1 = pd.Series(index=E[1:-2], dtype=float)

                # functions to be used in all loops
                rho_rovib_prod1 = pd.Series(f_rho_rovib_prod1(E), index = E)
                rho_non1 = pd.Series(convolve_f(f_rho_rovib_prod2, f_rhotrasl, E), index=E)
                ped_E = pd.Series(f_ped_E(E), index=E)

                for e in E[3:]:

                    # you are fixing e, so E1 can be only smaller than E
                    E1 = E[E <= e]
                    E_minus_E1 = E1[::-1]
                    # compute the convolution between rho1 and rhonon1
                    # int(rho1(E1)*rho_non1(E-E1)dE1)
                    rho1_E1 = rho_rovib_prod1[E1].values
                    rhonon1_E1 = rho_non1[E_minus_E1].values
                    rho1_rhonon1 = rho1_E1*rhonon1_E1
                    rhoE1tot = np.trapz(rho1_rhonon1, x=E1)
                    # print(rhoE1tot, np.trapz(rho1_rhonon1, x=E1)) #check
                    P_E1Etot = rho1_rhonon1/rhoE1tot
                    P_E1E[e][E1] = P_E1Etot
                # for each E1: extract P(E1;E) at different E and integrate
                # compute P(E1): you need to derive Pped[Eint]

                for E1 in E[1:-2]:
                    Evect = E[3:][E[3:] > E1]
                    P_E1Evect = P_E1E.loc[E1][Evect].values*ped_E[Evect].values
                    P_E1[E1] = np.trapz(P_E1Evect, x=Evect)

                norm_factor_P_E1 = np.trapz(
                    P_E1[E[1:-2]].values, x=E[1:-2])
                P_E1_norm = P_E1/norm_factor_P_E1
                ped_df_prod[P][T] = P_E1_norm

                P_E1_df[T][E[1:-2]] = P_E1_norm

                print(P, T, max(E),
                      P_E1_norm.idxmax(), P_E1_norm.max(), '\n')

            # write file - remove later
            P_E1_df = P_E1_df.reset_index()
            header_label = np.array(P_E1_df.columns, dtype=str)
            header_label[0] = 'E [kcal/mol]'
            labels = '\t\t'.join(header_label)
            np.savetxt('PE1_{}.txt'.format(P), P_E1_df.values,
                       delimiter='\t', header=labels, fmt='%1.2e')

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
