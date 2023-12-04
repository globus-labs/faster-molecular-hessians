"""Test for a MBTR-based energy model"""
import numpy as np
from pytest import fixture
from dscribe.descriptors import MBTR
from sklearn.linear_model import LinearRegression

from jitterbug.model.dscribe.globald import DScribeGlobalEnergyModel


@fixture()
def model(train_set):
    return DScribeGlobalEnergyModel(
        reference=train_set[0],
        descriptors=MBTR(
            species=["H", "C", "N", "O"],
            geometry={"function": "inverse_distance"},
            grid={"min": 0, "max": 1, "n": 100, "sigma": 0.1},
            weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
            periodic=False,
            normalization="l2",
        ),
        model=LinearRegression(),
        num_calculators=8
    )


def test_calculator(train_set, model):
    # Create then fit the model
    calcs = model.train(train_set)

    # Predict the energy (we should be close!)
    calc = calcs[0]
    test_atoms = train_set[0].copy()
    test_atoms.calc = calc
    energy = test_atoms.get_potential_energy()
    assert np.isclose(energy, train_set[0].get_potential_energy())

    # See if force calculation works
    forces = test_atoms.get_forces()
    assert forces.shape == (3, 3)  # At least make sure we get the right shape, values are iffy
    assert not np.isclose(forces, 0).all()


def test_hessian(train_set, model):
    """See if we can compute the Hessian"""

    # Run the fitting
    calcs = model.train(train_set)

    # Test the mean hessian function
    mean_hess = model.mean_hessian(calcs)
    assert mean_hess.shape == (9, 9), 'Wrong shape'
    assert np.isclose(mean_hess, mean_hess.T).all(), 'Not symmetric'

    # Test the sampling
    sampled_hess = model.sample_hessians(calcs, 128)
    assert all(np.isclose(hess, hess.T).all() for hess in sampled_hess)
    mean_sampled_hess = np.mean(sampled_hess, 0)
    assert np.isclose(np.diag(mean_sampled_hess), np.diag(mean_hess), atol=5).mean() > 0.5  # Make sure most agree
