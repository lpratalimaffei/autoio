""" MESS
"""

import mess_io.writer
from autorun._run import from_input_string

INPUT_NAME = 'mess.inp'
OUTPUT_NAMES = ('rate.out',)


# Specilialized runners
def torsions(script_str, run_dir, geo, hind_rot_str):
    """ Calculate the frequencies and ZPVES of the hindered rotors
        create a messpf input and run messpf to get tors_freqs and tors_zpes
    """

    # Write the MESSPF input file
    global_pf_str = mess_io.writer.global_pf_input(
        temperatures=(100.0, 200.0, 300.0, 400.0, 500),
        rel_temp_inc=0.001,
        atom_dist_min=0.6)
    dat_str = mess_io.writer.molecule(
        core=mess_io.writer.core_rigidrotor(geo, 1.0),
        freqs=(1000.0,),
        elec_levels=((0.0, 1.0),),
        hind_rot=hind_rot_str,
    )
    spc_str = mess_io.writer.species(
        spc_label='Tmp',
        spc_data=dat_str,
        zero_ene=0.0
    )
    input_str = '\n'.join([global_pf_str, spc_str]) + '\n'

    # Run the direct function and returns the output
    output_str = run_messpf(script_str, run_dir, input_str)

    # Read the torsional freqs and zpves
    tors_freqs = mess_io.reader.tors.analytic_frequencies(output_str)
    # tors_freqs = mess_io.reader.tors.grid_minimum_frequencies(output_str)
    tors_zpes = mess_io.reader.tors.zero_point_vibrational_energies(
        output_str)

    return tors_freqs, tors_zpes


def run_messpf(script_str, run_dir, input_str):
    """ Runs messpf with default input and output filenames
        :param input_str: string block which constitutes the input file
        :type input_str: str
        :param script_str: string of bash script that contains
            execution instructions electronic structure job
        :type script_str: str
        :param run_dir: name of directory to run electronic structure job
        :type run_dir: str
        :returns output_str: string with the content of the output file
        :rtype: str
    """
    input_name = 'pf.inp'
    output_name = 'pf.log'
    output_strs = direct(script_str, run_dir, input_str,
                         aux_dct=None,
                         input_name=input_name,
                         output_names=(output_name,))
    output_str = output_strs[0]

    return output_str


def direct(script_str, run_dir, input_str, aux_dct=None,
           input_name=INPUT_NAME,
           output_names=OUTPUT_NAMES):
    """
        :param input_str: string block which constitutes the input file
        :type input_str: str
        :param aux_dct: auxiliary input strings dict[name: string]
        :type aux_dct: dict[str: str]
        :param script_str: string of bash script that contains
            execution instructions electronic structure job
        :type script_str: str
        :param run_dir: name of directory to run electronic structure job
        :type run_dir: str
    """

    output_strs = from_input_string(
        script_str, run_dir, input_str,
        aux_dct=aux_dct,
        input_name=input_name,
        output_names=output_names)

    return output_strs
