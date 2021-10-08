"""
Analyze the extent of prompt dissociation
for a single exothermic reaction with successive decomposition
"""
import os
import numpy as np
import json
import argparse
import mess_io
from autofile.io_ import read_file
from mess_io.writer import rename_ktp_dct
from mess_io.writer import fit_ktp_dct


##################################################################

# WHEN INTEGRATED IN MECHDRIVER: REMOVE, MUST BE AUTOMATIC
# BF THRESHOLD AS INPUT.. WHERE?
# name of the prompt dissociating radical in both PESs
rad_name = ['HCO', 'W1']
bf_threshold = 0.1  # minimum 10% of BF to include the species in the products

#modeltype_list = ['beta_phi1a', 'beta_phi2a','beta_phi3a','equip_phi','equip_simple','rovib_dos']  # type of model
modeltype_list = ['beta_phi1a',]  # type of model

def _read_json(path, file_name):
    ''' read a json file with dictionary defined
    '''

    json_path = os.path.join(path, file_name)
    if os.path.exists(json_path):
        with open(json_path, 'r') as fobj:
            json_dct = json.load(fobj)
    else:
        json_dct = None

    return json_dct


# INPUT READING. NB THE I/O WILL BE EDITED AUTOMATICALLY UPON INTEGRATION IN MECHDRIVER
CWD = os.getcwd()
# to launch this: python prompt.py &; default stuff will be searched for automatically.
# Parse the command line
PAR = argparse.ArgumentParser()
PAR.add_argument('-iped', '--pedinput', default='me_ktp_ped.inp',
                 help='MESS input name (me_ktp_ped.inp)')
PAR.add_argument('-oped', '--pedoutput', default='rate_ped.out',
                 help='MESS ouput name (rate_ped.out)')
PAR.add_argument('-opedmicro', '--pedoutputmicro', default='ke_ped.out',
                 help='MESS microcanonical ouput name (ke_ped.out)')
PAR.add_argument('-ihot', '--hotinput', default='me_ktp_hoten.inp',
                 help='MESS hoten input name (me_ktp_hoten.inp)')
PAR.add_argument('-ohot', '--hotoutput', default='me_ktp_hoten.log',
                 help='MESS hoten log name (me_ktp_hoten.log)')
PAR.add_argument('-l', '--label', default='label.inp',
                 help='label dct name (label.inp)')

OPTS = vars(PAR.parse_args())  # it's a dictionary

# READ INITIAL FILES AND LABEL DICTIONARY
label_dct = _read_json(CWD, OPTS['label'])

############ DO NOT MODIFY ##################
# SECONDO ME CONVIENE ALLA FINE METTERE TUTTO IN UNA CLASSE
# OPERATIONS
# 0. EXTRACT INPUT INFORMATION
me_ktp_inp = read_file(os.path.join(CWD, OPTS['pedinput']))
species_blocks_ped = mess_io.reader.get_species(me_ktp_inp)
T_lst, _ = mess_io.reader.rates.temperatures(me_ktp_inp, mess_file='inp')
P_lst, _ = mess_io.reader.rates.pressures(me_ktp_inp, mess_file='inp')
P_lst = P_lst[:-1]  # drop the last element in the pressure list ('high')
pedspecies, pedoutput = mess_io.reader.ped.ped_names(me_ktp_inp)
reacs, prods = pedspecies


# ktp dictionary
mess_out = read_file(os.path.join(CWD, OPTS['pedoutput']))
ktp_dct = mess_io.reader.rates.ktp_dct(
    mess_out, reacs, prods)
energies_sp, energies_ts = mess_io.reader.rates.energies(mess_out)
_, E_BW = mess_io.reader.rates.barriers(energies_ts, energies_sp, reacs, prods)

print(energies_sp, energies_ts, reacs, prods)


# 2. READ THE HOTENERGIES OUTPUT
hot_str = read_file(os.path.join(CWD, OPTS['hotinput']))
hot_str_output = read_file(os.path.join(CWD, OPTS['hotoutput']))
species_blocks = mess_io.reader.get_species(hot_str)
T_lst_hot, _ = mess_io.reader.rates.temperatures(hot_str, mess_file='inp')
P_lst_hot, _ = mess_io.reader.rates.pressures(hot_str, mess_file='inp')
# drop the last element in the pressure list ('high')
P_lst_hot = P_lst_hot[:-1]
hotspecies = mess_io.reader.hotenergies.get_hot_names(hot_str)
hoten_dct = mess_io.reader.hotenergies.extract_hot_branching(
    hot_str_output, hotspecies, list(species_blocks.keys()), T_lst_hot, P_lst_hot)

# print(ktp_dct)
# 1. READ THE PEDOUTPUT file and reconstruct the energy distribution
pedoutput_str = read_file(os.path.join(CWD, pedoutput))
ped_df = mess_io.reader.ped.get_ped(pedoutput_str, pedspecies, energies_sp)
dof_info = mess_io.reader.ped.ped_dof_MW(species_blocks_ped[prods])

# 1b. READ THE ke_ped.out file and extract the energy density of each fragment
ke_ped_out = read_file(os.path.join(CWD, OPTS['pedoutputmicro']))
dos_df = mess_io.reader.rates.dos_rovib(ke_ped_out)

# derive P_E1

for modeltype in modeltype_list:
    ped_df_prod1 = mess_io.reader.ped.ped_prod1(
        ped_df, rad_name[0], modeltype, dos_df=dos_df, dof_info=dof_info, E_BW=E_BW)


    # 3. DERIVE T,P DEPENDENT PRODUCT BRANCHING FRACTIONS and decide which species to keep
    bf_tp_df = mess_io.reader.bf.bf_tp_df_full(rad_name, ped_df_prod1, hoten_dct)
    bf_tp_dct = mess_io.reader.bf.bf_tp_dct_filter(
        bf_tp_df, bf_threshold, modeltype, T_all=T_lst)
    print(bf_tp_dct)

# 4. DO ARRHENIUS FITS FOR THE SELECTED BFs
rxn_ktp_dct = mess_io.reader.bf.merge_bf_rates(bf_tp_dct, ktp_dct)

# rename the ktp dictionary with appropriate reaction names
rxn_ktp_dct = rename_ktp_dct(rxn_ktp_dct, pedspecies, label_dct)
# print(rxn_ktp_dct)

fitted_dct = fit_ktp_dct(rxn_ktp_dct, CWD)
print(fitted_dct)
#
