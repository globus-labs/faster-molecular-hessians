"""Utility functions"""
from typing import Optional


from ase.calculators.calculator import Calculator
from ase.calculators.mopac import MOPAC
from ase.calculators.psi4 import Psi4

mopac_methods = ['pm7']
"""List of methods for which we will use MOPAC"""


def make_calculator(method: str, basis: Optional[str], **kwargs) -> Calculator:
    """Make an ASE calculator that implements a desired method.

    This function will select the appropriate quantum chemistry code depending
    on the method name using the following rules:

    1. Use MOPAC if the method is PM7.
    2. Use Psi4 otherwise

    Any keyword arguments are passed to the calculator

    Args:
        method: Name of the quantum chemistry method
        basis: Basis set name, if appropriate
    Returns:
        Calculator defined according to the user's settings
    """

    if method in mopac_methods:
        if not (basis is None or basis == "None"):
            raise ValueError(f'Basis must be none for method: {method}')
        return MOPAC(method=method, command='mopac PREFIX.mop > /dev/null')
    else:
        return Psi4(method=method, basis=basis, **kwargs)
