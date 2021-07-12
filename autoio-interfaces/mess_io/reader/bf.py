""" Extract branching fractions of the products from
    hotenergies and pedspecies
    Fit and write the rates
"""

import sys
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd


def bf_tp_df_full(prod_name, ped_df_prod, hoten_dct):
    """ Build a branching fractions dictionary as a function of temeprature and pressure
        containing the BFs of each product of the PES
        :param prod_name: name of the product in pedspecies and in hotenergies
        :type prod_name: list(str)
        :param ped_df: dataframe[P][T] with the Series of energy distrib [en: prob(en)]
        :type ped_df: dataframe(series(float))
        :param hoten_dct: hot branching fractions for hotspecies
        :type hoten_dct: dct{hotspecies: df[P][T]:df[allspecies][energies]}
        :return bf_tp_df: branching fractions at T,P for each product for the selected hotspecies
        :rtype: dataframe of series df[P][T]:series[species], dataframe(series(float))
    """
    # preprocessing: extract data and expand range of pressure if needed
    p1_hot = prod_name[1]
    hoten_df_p1 = hoten_dct[p1_hot]
    # sort indexes
    ped_df_prod = ped_df_prod.sort_index()
    hoten_df_p1 = hoten_df_p1.sort_index()
    # compare T,P:
    T_ped, P_ped = [ped_df_prod.index, ped_df_prod.columns]
    T_hot, P_hot = [hoten_df_p1.index, hoten_df_p1.columns]
    print(P_ped, P_hot)
    # check that the T of T_hot are at least as many as those of T_ped
    # if they're not, it doesn't make sense to continue
    if not all(T_ped_i in T_hot for T_ped_i in T_ped):
        print('*Error: temperature range in HOTenergies does not cover the full range')
        sys.exit()

    # if in P_ped not all pressures of hoten are available: extend the range of pressures
    # ex.: for H abstractions, they will be pressure independent but probably the successive decomposition is not
    if not all(P_hot_i in P_ped for P_hot_i in P_hot):
        print('*Warning: P range of PedOutput smaller than HOTenergies: \n')
        print('Energy distribution at other pressure approximated from available values \n')
        ped_df_prod = extend_dct_withP(P_hot, P_ped, ped_df_prod)

    # check that the pressures of P_ped are contained in hot energies
    # if they are not: extend the pressure range assuming the behavior is the same
    if not all(P_ped_i in P_hot for P_ped_i in P_ped):
        print('Warning: P range of HOTenergies smaller than PEDoutput: \n')
        print(
            'Energy distribution at other pressures approximated from available values \n')
        hoten_df_p1 = extend_dct_withP(P_ped, P_hot, hoten_df_p1)

    # compute branching fractions
    # derive keys for dictionaries and complete set of T,P (should be identical)
    # T_ped contains the desired T range; T_hot may have a larger range
    P_lst, T_lst = [hoten_df_p1.columns, T_ped]
    allspecies = hoten_df_p1[P_lst[0]][T_lst[0]].columns  # extract all species
    # for each T, P: compute BF
    bf_tp_df = pd.DataFrame(index=T_lst, columns=P_lst, dtype=object)
    for T in T_lst:
        for P in P_lst:
            # extract ped and hoten by increasing index
            ped = ped_df_prod[P][T].sort_index()
            hoten = hoten_df_p1[P][T].sort_index().index
            # reduce the energy range where you have some significant ped probability
            ped = ped[ped > 0.001]
            hoten = hoten[hoten <= ped.index[-1]]
            if len(hoten) >= 4:
                E_vect = ped.index
                ped_vect = ped.values
                # Series to allocate
                bf_series = pd.Series(0, index=allspecies, dtype=float)
                for sp in allspecies:
                    hoten_sp = hoten_df_p1[P][T][sp][hoten]
                    f_hoten = interp1d(
                        hoten_sp.index, hoten_sp.values, kind='cubic', fill_value='extrapolate')
                    hoten_vect = f_hoten(E_vect)
                    # recompute in an appropriate range
                    bf_series[sp] = np.trapz(ped_vect*hoten_vect, x=E_vect)

                # renormalize for all species and put in dataframe
                bf_tp_df[P][T] = bf_series/np.sum(bf_series.values)

        # if any nan: delete column
        if any(bf_tp_df.loc[T].isnull()):
            bf_tp_df = bf_tp_df.drop(index=[T])

    return bf_tp_df


def bf_tp_dct_filter(bf_tp_df, bf_threshold, model, T_all=None):
    """
    Converts the dataframe of hot branching fractions to dictionary and excludes invalid BFs
        :param bf_tp_df: branching fractions at T,P for each product for the selected hotspecies
        :type bf_tp_df: dataframe of series df[P][T]:series[species], dataframe(series(float))
        :param bf_threshold: threshold BF: filter out all species with lower BFs
        :param T_all: vector with all temperatures (BF extended or calculated in the desired range)
        :type T_all: list
        :return bf_tp_dct: branching fractions at T,P for each product
        :rtype: dct{species: {pressure: (T, BF)}} - same type as ktp dct
    """
    # get species
    T_lst, P_lst = bf_tp_df.index, bf_tp_df.columns
    if not T_all:
        T_all = T_lst
    allspecies = bf_tp_df.iloc[0, 0].index
    bf_tp_dct_out = {}
    # fill the dictionary:
    for sp in allspecies:
        N_data_highenough = 0
        bf_tp_dct_i = {}
        bf_df_sp_i = pd.DataFrame(
            np.zeros((len(T_lst), len(P_lst))), index=T_lst, columns=P_lst)

        for P in P_lst:
            bf_T = []
            T_new = []
            for T in T_lst:
                bf = bf_tp_df[P][T][sp]
                if bf >= 1e-10:  # avoid too small values
                    T_new.append(T)
                    bf_T.append(bf)
                    bf_df_sp_i[P][T] = bf
                # check if values is high enough
                if bf >= bf_threshold:
                    N_data_highenough += 1
                # condition on BF and temperature
            if bf_T:
                # extend the range of the fits to the full range
                # f_bf = interp1d(
                #     T_new, bf_T, kind='cubic', fill_value='extrapolate', bounds_error=False)
                # bf_T_all = f_bf(T_all)
                # bf_tp_dct_i[P] = (np.array(T_all), np.array(bf_T_all))
                bf_tp_dct_i[P] = (np.array(T_new), np.array(bf_T))

        if N_data_highenough > 0:
            bf_tp_dct_out[sp] = bf_tp_dct_i

        # write file with the BFs
        bf_df_sp_i = bf_df_sp_i.reset_index()
        header_label = np.array(bf_df_sp_i.columns, dtype=str)
        header_label[0] = 'T [K]'
        labels = '\t\t'.join(header_label)
        np.savetxt('bf_{}_{}.txt'.format(sp, model), bf_df_sp_i.values,
                   delimiter='\t', header=labels, fmt='%1.2e')

    return bf_tp_dct_out


def merge_bf_rates(bf_tp_dct, ktp_dct):
    """ Read the branching fractions of the products and derive final rate constants
    """
    # drop "high" from ktp_dct
    if 'high' in ktp_dct.keys():
        ktp_dct.pop('high')

    # get all species and preallocate new dct
    all_species = bf_tp_dct.keys()
    new_ktp_dct = dict.fromkeys(all_species)

    # loop over all species
    for sp, bf_tp_dct_sp in bf_tp_dct.items():

        P_all = list(bf_tp_dct_sp.keys())
        P_ktp = list(ktp_dct.keys())
        # extend ktp_dct to the full range of pressures
        if not all(P in P_ktp for P in P_all):
            print('Warning: P range of ktp dictionary extended to match BF: \n')
            print(
                'values at other pressures approximated from available values \n')
            ktp_dct = extend_dct_withP(P_all, P_ktp, ktp_dct)

        bf_ktp_dct = dict.fromkeys(P_all)
        for P in P_all:
            T_bf = bf_tp_dct_sp[P][0]
            T_ktp = np.array(ktp_dct[P][0])
            # choose the smallest vector
            T_common = T_bf[[T_bf_i in T_ktp for T_bf_i in T_bf]]
            # reconstruct appropriate datasets
            bf_series = pd.Series(bf_tp_dct_sp[P][1], index=T_bf)
            ktp_series = pd.Series(np.array(ktp_dct[P][1]), index=T_ktp)
            # reconstruct k and allocate
            kvals = bf_series[T_common]*ktp_series[T_common]
            bf_ktp_dct[P] = (T_common, kvals)
            # check that the temperatures correspond
            # if any(T_all != np.array(ktp_dct[P][0])):
            #     print('*Error: T vector of the BFs and of ktp dct rates are different')
            #    sys.exit()
            # else:
            #     kvals = bf_tp_dct_sp[P][1]*np.array(ktp_dct[P][1])
            #     bf_ktp_dct[P] = (T_all, kvals)

        # allocate the new dictionary
        new_ktp_dct[sp] = bf_ktp_dct

    return new_ktp_dct


def extend_dct_withP(P_all, P_red, dct_toextend):
    """ Takes a dictionary with pressure keys and extends it:
        approximate the values at missing pressures with available values
        :param P_all: all pressures
        :type P_all: list
        :param dct_toextend: dictionary with P_reduced as keys
        :type dct_toextend: dictionary or dataframe
        :return dct_extended: dictionary with P_all as keys
        :rtype: dictionary or dataframe
    """

    for P in P_all:
        if P not in P_red:
            # approximate pressure: provides the minimum difference with P_ped_i
            P_approx = P_red[np.argmin(
                [np.log(abs(P_red_i-P)) for P_red_i in P_red])]
            # extend original dataframe
            dct_toextend[P] = dct_toextend[P_approx]

    return dct_toextend
