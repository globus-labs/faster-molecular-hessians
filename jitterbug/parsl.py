"""Wrappers for functions compatible with the Parsl workflow engine"""
import os
from tempfile import TemporaryDirectory
from typing import Optional
from pathlib import Path

import ase

from jitterbug.utils import make_calculator, write_to_string


def get_energy(atoms: ase.Atoms, method: str, basis: Optional[str], scratch_dir: Optional[str] = None, **kwargs) -> str:
    """Compute the energy of an atomic structure

    Keyword arguments are passed to :meth:`make_calculator`.

    Args:
        atoms: Structure to evaluate
        method: Name of the method to use (e.g., B3LYP)
        basis: Basis set to use (e.g., cc-PVTZ)
        scratch_dir: Path to the scratch directory.
    Returns:
        Atoms record serialized with the energy and any other data produced by the calculator
    """

    # Make a temporary directory
    start_dir = Path.cwd()
    tmp = TemporaryDirectory(dir=scratch_dir, prefix='jitterbug_')
    try:
        os.chdir(tmp.name)
        calc = make_calculator(method, basis, directory=tmp.name, **kwargs)
        atoms.calc = calc
        atoms.get_potential_energy()
        return write_to_string(atoms, 'json')
    finally:
        os.chdir(start_dir)
        tmp.cleanup()
