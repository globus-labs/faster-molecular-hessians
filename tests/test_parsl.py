from ase.io import read

from jitterbug.parsl import get_energy


def test_energy(xyz_path):
    atoms = read(xyz_path)
    get_energy(atoms, 'pm7', None)
