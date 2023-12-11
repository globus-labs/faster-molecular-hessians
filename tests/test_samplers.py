from pytest import mark
from ase.io import read

from jitterbug.sampler import methods


@mark.parametrize('method', ['simple'])
def test_random(method, xyz_path):
    """Make sure we get the same structures each time"""

    atoms = read(xyz_path)
    sampler = methods[method]()

    # Generate two batches
    samples_1 = sampler.produce_structures(atoms, 4)
    samples_2 = sampler.produce_structures(atoms, 8)

    for a1, a2 in zip(samples_1, samples_2):
        assert a1 == a2
