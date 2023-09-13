"""Test for a MBTR-based energy model"""
import numpy as np

from jitterbug.model.mbtr import MBTRCalculator, MBTREnergyModel


def test_model(train_set):
    # Create then fit the model
    calc = MBTRCalculator()
    calc.train(train_set)

    # Predict the energy (we should be close!)
    test_atoms = train_set[0].copy()
    test_atoms.calc = calc
    energy = test_atoms.get_potential_energy()
    assert np.isclose(energy, train_set[0].get_potential_energy())

    # See if force calculation works
    forces = test_atoms.get_forces()
    assert forces.shape == (3, 3)  # At least make sure we get the right shape, values are iffy


def test_hessian(train_set):
    """See if we can compute the Hessian"""
    calc = MBTRCalculator()
    model = MBTREnergyModel(calc, train_set[0])

    # Run the fitting
    hess_model = model.train(train_set)
    hess = model.mean_hessian(hess_model)
    assert hess.shape == (9, 9)
