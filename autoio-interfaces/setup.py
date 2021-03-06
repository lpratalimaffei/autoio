""" Install Interfaces to MESS, CHEMKIN, VaReCoF, ProjRot, and ThermP
"""

from distutils.core import setup

setup(
    name="autoio-interfaces",
    version="0.7.0",
    packages=['mess_io',
              'mess_io.writer',
              'mess_io.reader',
              'projrot_io',
              'polyrate_io',
              'varecof_io',
              'varecof_io.writer',
              'varecof_io.reader',
              'chemkin_io',
              'chemkin_io.writer',
              'chemkin_io.parser',
              'onedmin_io',
              'pac99_io',
              'intder_io',
              'thermp_io',
              'autorun',
              'elstruct',
              'elstruct.writer',
              'elstruct.reader',
              'elstruct.writer._psi4',
              'elstruct.reader._psi4',
              'elstruct.writer._cfour2',
              'elstruct.reader._cfour2',
              'elstruct.writer._gaussian09',
              'elstruct.reader._gaussian09',
              'elstruct.writer._gaussian16',
              'elstruct.reader._gaussian16',
              'elstruct.writer._mrcc2018',
              'elstruct.reader._mrcc2018',
              'elstruct.writer._nwchem6',
              'elstruct.reader._nwchem6',
              'elstruct.writer._orca4',
              'elstruct.reader._orca4',
              'elstruct.writer._molpro2015',
              'elstruct.reader._molpro2015'],
    package_dir={
        'mess_io': 'mess_io',
        'projrot_io': 'projrot_io',
        'varecof_io': 'varecof_io',
        'polyrate_io': 'polyrate_io',
        'chemkin_io': 'chemkin_io',
        'onedmin_io': 'onedmin_io',
        'pac99_io': 'pac99_io',
        'intder_io': 'intder_io',
        'elstruct': 'elstruct',
        'thermp_io': 'thermp_io',
        'autorun': 'autorun'},
    package_data={
        'mess_io': ['writer/templates/sections/*.mako',
                    'writer/templates/sections/energy_transfer/*.mako',
                    'writer/templates/sections/monte_carlo/*.mako',
                    'writer/templates/sections/reaction_channel/*.mako',
                    'writer/templates/species/*.mako',
                    'writer/templates/species/info/*.mako',
                    'tests/data/*.txt'],
        'intder_io': ['templates/*.mako'],
        'projrot_io': ['templates/*.mako',
                       'tests/data/*.txt'],
        'polyrate_io': ['tests/templates/*.mako',
                        'templates/*.mako'],
        'onedmin_io': ['tests/templates/*.mako',
                       'templates/*.mako'],
        'varecof_io': ['writer/templates/*.mako',
                       'tests/data/*.txt'],
        'thermp_io': ['templates/*.mako'],
        'elstruct': ['writer/_psi4/templates/*.mako',
                     'writer/_cfour2/templates/*.mako',
                     'writer/_gaussian09/templates/*.mako',
                     'writer/_gaussian16/templates/*.mako',
                     'writer/_mrcc2018/templates/*.mako',
                     'writer/_nwchem6/templates/*.mako',
                     'writer/_orca4/templates/*.mako',
                     'writer/_molpro2015/templates/*.mako'],
        'autorun': ['tests/data/*', 'aux/*'],
        'chemkin_io': ['tests/data/*.txt', 'tests/data/*.csv']}
)
