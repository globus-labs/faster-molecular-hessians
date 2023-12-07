"""Wrappers for functions compatible with the Parsl workflow engine"""
from typing import Optional

import ase

from jitterbug.utils import make_calculator, write_to_string


def get_energy(atoms: ase.Atoms, method: str, basis: Optional[str], **kwargs) -> str:
    """Compute the energy of an atomic structure

    Keyword arguments are passed to :meth:`make_calculator`.

    Args:
        atoms: Structure to evaluate
        method: Name of the method to use (e.g., B3LYP)
        basis: Basis set to use (e.g., cc-PVTZ)
    Returns:
        Atoms record serialized with the energy and any other data produced by the calculator
    """

    calc = make_calculator(method, basis, **kwargs)
    atoms.calc = calc
    atoms.get_potential_energy()
    return write_to_string(atoms, 'json')
