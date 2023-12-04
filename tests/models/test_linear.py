import numpy as np
from ase.build import molecule
from ase.vibrations import VibrationsData

from jitterbug.model.linear import get_model_inputs, HarmonicModel
from jitterbug.model.linear_internals import HarmonicModel as ICHarmonicModel


def test_disp_matrix():
    reference = molecule('H2')
    atoms = reference.copy()

    # With a single displacement only the first term should be nonzero
    atoms.positions[0, 0] += 0.1
    disp_matrix = get_model_inputs(atoms, reference)
    assert disp_matrix.shape == (27,)  # 6 linear, 21 harmonic terms
    assert (disp_matrix != 0).sum() == 2  # One linear and one harmonic term
    assert np.isclose(disp_matrix[0], 0.1)  # Linear terms
    assert np.isclose(disp_matrix[6], 0.01 / 2)  # Harmonic terms

    # With two displacements, there should be 3 nonzero terms
    atoms.positions[1, 0] += 0.05
    disp_matrix = get_model_inputs(atoms, reference)
    assert (disp_matrix != 0).sum() == 2 + 3
    assert np.isclose(disp_matrix[[0, 3]], [0.1, 0.05]).all()  # Linear terms
    assert np.isclose(disp_matrix[6], 0.01 / 2)  # (Atom 0, x) * (Atom 0, x)
    assert np.isclose(disp_matrix[6 + 3], 0.1 * 0.05)  # (Atom 0, x) * (Atom 1, x) * 2 (harmonic)
    assert np.isclose(disp_matrix[6 + 6 + 5 + 4], 0.0025 / 2)  # (Atom 1, x) * (Atom 1, x)


def test_linear_model(train_set):
    # The first atom in the set should have forces
    reference = train_set[0]
    assert reference.get_forces().max() < 0.01

    # Fit the model
    model = HarmonicModel(reference)
    hessian_model = model.train(train_set)
    assert hessian_model.coef_.shape == (54,)

    # Get the mean hessian
    hessian = model.mean_hessian(hessian_model)
    assert hessian.shape == (9, 9)

    # Sample the Hessians, at least make sure the results are near correct
    hessians = model.sample_hessians(hessian_model, num_samples=32)
    assert len(hessians) == 32
    assert np.isclose(hessians[0], hessians[0].T).all()

    # Create a vibration data object
    vib_data = VibrationsData.from_2d(reference, hessians[0])
    zpe = vib_data.get_zero_point_energy()
    print(zpe)
    assert np.isclose(zpe, 0.63, atol=0.3)  # Make sure it's _close_


def test_linear_internal_model(train_set):
    # The first atom in the set should have forces
    reference = train_set[0]
    assert reference.get_forces().max() < 0.01

    # Fit the model
    model = ICHarmonicModel(reference)
    hessian_model = model.train(train_set)
    assert hessian_model.coef_.shape == (9,)

    # Get the mean hessian
    hessian = model.mean_hessian(hessian_model)
    assert hessian.shape == (9, 9)

    # Sample the Hessians, at least make sure the results are near correct
    hessians = model.sample_hessians(hessian_model, num_samples=32)
    assert len(hessians) == 32
    assert np.isclose(hessians[0], hessians[0].T).all()

    # Create a vibration data object
    vib_data = VibrationsData.from_2d(reference, hessians[0])
    zpe = vib_data.get_zero_point_energy()
    print(zpe)
    assert np.isclose(zpe, 0.63, atol=0.3)  # Make sure it's _close_
