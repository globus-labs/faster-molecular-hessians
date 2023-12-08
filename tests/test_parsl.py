from pathlib import Path

from ase.io import read
from pytest import mark

from jitterbug.parsl import get_energy, load_configuration
from jitterbug.utils import read_from_string


def test_energy(xyz_path):
    mopac_out = Path('mopac.out')
    mopac_out.unlink(missing_ok=True)

    atoms = read(xyz_path)
    atoms_msg = get_energy(atoms, 'pm7', None)
    new_atoms = read_from_string(atoms_msg, 'extxyz')
    assert 'energy' in new_atoms.calc.results

    assert not mopac_out.exists()


@mark.parametrize('method,basis,is_psi4', [('hf', 'sto-3g', True), ('xtb', None, False), ('pm7', None, False)])
def test_radical(file_dir, method, basis, is_psi4):
    """Test radical computation with Psi4"""
    atoms = read(file_dir / 'radical.extxyz')
    mult = int(atoms.get_initial_magnetic_moments().sum() + 1)
    if is_psi4:
        atoms.set_initial_magnetic_moments([0] * len(atoms))

    get_energy(atoms, method, basis, multiplicity=mult)


def test_load(file_dir):
    config, workers, options = load_configuration(file_dir / 'example_config.py')
    assert config.executors[0].max_workers == 1
    assert workers == 1
    assert options == {}
