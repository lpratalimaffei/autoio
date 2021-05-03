"""
  Centralized autorun functions
"""

# Single Program Runners
from autorun import mess
from autorun import onedmin
from autorun import pac99
from autorun import polyrate
from autorun import projrot
from autorun import thermp

# MultiProgram Runners
from autorun._multiprog import projected_frequencies
from autorun._multiprog import thermo

# Useful Running Functions
from autorun._script import SCRIPT_DCT
from autorun._run import from_input_string
from autorun._run import run_script
from autorun._run import write_input
from autorun._run import read_output


__all__ = [
    # Single Program Runners
    'mess',
    'onedmin',
    'pac99',
    'polyrate',
    'projrot',
    'thermp',
    # MultiProgram Runners
    'projected_frequencies',
    'thermo',
    # Useful Running Functions
    'SCRIPT_DCT',
    'from_input_string',
    'run_script',
    'write_input',
    'read_output'
]