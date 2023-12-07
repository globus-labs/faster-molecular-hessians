from pathlib import Path

from ase.io import read

from jitterbug.parsl import get_energy
from jitterbug.utils import read_from_string


def test_energy(xyz_path):
    mopac_out = Path('mopac.out')
    mopac_out.unlink(missing_ok=True)

    atoms = read(xyz_path)
    atoms_msg = get_energy(atoms, 'pm7', None)
    new_atoms = read_from_string(atoms_msg, 'json')
    assert 'energy' in new_atoms.calc.results

    assert not mopac_out.exists()
