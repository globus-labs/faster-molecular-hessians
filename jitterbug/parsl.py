"""Wrappers for functions compatible with the Parsl workflow engine"""
from typing import Optional

import ase

from jitterbug.utils import make_calculator


def get_energy(atoms: ase.Atoms, method: str, basis: Optional[str], **kwargs) -> float:
    """Compute the energy of an atomic structure

    Keyword arguments are passed to :meth:`make_calculator`.

    Args:
        atoms: Structure to evaluate
        method: Name of the method to use (e.g., B3LYP)
        basis: Basis set to use (e.g., cc-PVTZ)
    Returns:
        Energy (units: eV)
    """

    calc = make_calculator(method, basis, **kwargs)
    return calc.get_potential_energy(atoms)
