"""Utility functions"""
from typing import Optional
from io import StringIO

from ase.calculators.calculator import Calculator
from ase.calculators.mopac import MOPAC
from ase.calculators.psi4 import Psi4
from ase import Atoms, io

mopac_methods = ['pm7']
"""List of methods for which we will use MOPAC"""


def make_calculator(method: str, basis: Optional[str], multiplicity: int = 1,
                    code: Optional[str] = None, **kwargs) -> Calculator:
    """Make an ASE calculator that implements a desired method.

    This function will select the appropriate quantum chemistry code depending
    on the method name using the following rules:

    1. Use the code named in the ``code`` argument
    2. Use MOPAC if the method is PM7.
    3. Use xTB if the method is xTB.
    4. Use Psi4 otherwise

    Any keyword arguments are passed to the calculator

    Args:
        method: Name of the quantum chemistry method
        basis: Basis set name, if appropriate
        multiplicity: Default multiplicity for the computation
        code: Which code to use
    Returns:
        Calculator defined according to the user's settings
    """

    # Determine which code to use
    if code is None:
        if method in mopac_methods:
            if not (basis is None or basis.lower() == "none"):
                raise ValueError(f'Basis must be none for method: {method}')
            code = 'mopac'
        elif method == 'xtb':
            code = 'xtb'
        else:
            code = 'psi4'

    # Make the appropriate calculator
    if code == 'mopac':
        return MOPAC(method=method, command='mopac PREFIX.mop > /dev/null')
    elif code == 'xtb':
        from xtb.ase.calculator import XTB
        return XTB()
    elif code == 'psi4':
        return Psi4(method=method, basis=basis,
                    multiplicity=multiplicity,
                    reference='rhf' if multiplicity == 1 else 'uhf',
                    **kwargs)
    elif code == 'exachem':
        from exachem.calc import ExaChem
        return ExaChem(method=method, basisset=basis, **kwargs)
    else:
        raise ValueError(f'Code not supported: {code}')


# Taken from ExaMol
def write_to_string(atoms: Atoms, fmt: str, **kwargs) -> str:
    """Write an ASE atoms object to string

    Args:
        atoms: Structure to write
        fmt: Target format
        kwargs: Passed to the write function
    Returns:
        Structure written in target format
    """

    out = StringIO()
    atoms.write(out, fmt, **kwargs)
    return out.getvalue()


def read_from_string(atoms_msg: str, fmt: str) -> Atoms:
    """Read an ASE atoms object from a string

    Args:
        atoms_msg: String format of the object to read
        fmt: Format (cannot be autodetected)
    Returns:
        Parsed atoms object
    """

    out = StringIO(str(atoms_msg))  # str() ensures that Proxies are resolved
    return io.read(out, format=fmt)
