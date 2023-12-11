from pytest import mark
from ase.io import read
import numpy as np

from jitterbug.sampler import methods


@mark.parametrize('method', ['uniform'])
def test_random(method, xyz_path):
    """Make sure we get the same structures each time"""

    atoms = read(xyz_path)
    sampler = methods[method]()
    assert sampler.name.startswith(method)

    # Generate two batches
    samples_1 = sampler.produce_structures(atoms, 4)
    samples_2 = sampler.produce_structures(atoms, 8)

    for a1, a2 in zip(samples_1, samples_2):
        assert np.isclose(a1.positions, a2.positions).all()
