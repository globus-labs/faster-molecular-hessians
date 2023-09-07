"""Test for a MBTR-based energy model"""
import numpy as np

from jitterbug.model.mbtr import MBTRCalculator


def test_model(train_set):
    # Create then fit the model
    calc = MBTRCalculator()
    calc.train(train_set)

    # Predict the energy (we should be close!)
    energy = calc.get_potential_energy(train_set[0])
    assert np.isclose(energy, train_set[0].get_potential_energy())
